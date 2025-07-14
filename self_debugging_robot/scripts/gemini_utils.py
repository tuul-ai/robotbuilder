import json
from diffusers.utils import load_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
import cv2
from tqdm.notebook import tqdm
from typing import List, Optional, Dict, Union
import os
import torch
import time
from collections import deque

from google import genai
from google.genai import types
import base64
from io import BytesIO
from .image_utils import tensor_to_pil, reorder_tensor_dimensions
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.0-flash"  # Use Gemini 2.0 Flash for 3D capabilities
PRO_MODEL_ID ='gemini-2.0-pro-exp-02-05'

def parse_json(response_text):
    # Parsing out the markdown fencing
    # Find JSON content within triple backticks if present
    try:
        # Remove markdown code block formatting, if present
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response_text}")
        return []
    
def analyze_scene(img, prompt=None) -> str:
    """Prompts Gemini 2.0 Flash for scene analysis (3D bounding boxes, orientation)."""
    if prompt is None:
        prompt = """Describe the scene from the top view. Focus on objects, positions, and spatial relationships. 
        Point to no more than 10 items in the image. Include following items in the analysis: Robot arm, container, and Lego bricks.
        The answer should follow the json format: [{"point": <point>, "label": <label1>, "description": <description>}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.
        """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[img, prompt],
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text
    
def analyze_multi_view(img_0, img_1, prompt=None, context=None) -> str:
    if prompt is None:
        prompt = """Given the top view image and analysis, get additional details from the front view image. 
        Provide a combined analysis of the top and front views. Focus on heights, occlusions, and depth relationships.
        The answer should follow the json format: [{"point": <point>, "label": <label1>, "description": <description>}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.
        """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[img_0, prompt, context, img_1],
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text

def get_summary(analysis_data: List[Dict], prompt_template: str) -> str:
    """Prompts Gemini (text model) for a summary of the analysis."""
    # Flatten the analysis data into a single string (for text-based Gemini)
    combined_analysis = "\n".join([str(episode) for episode in analysis_data])

    try:
        response = client.models.generate_content(
            model=PRO_MODEL_ID,  # Use this or something similar for text generation
            contents=[prompt_template.format(combined_analysis)],
            config=types.GenerateContentConfig(temperature=0.2, top_p=0.8)
        )
    except genai.ServerError:  # Catch the ServerError
        response = client.models.generate_content(
            model=MODEL_ID,  # Fallback to MODEL_ID
            contents=[prompt_template.format(combined_analysis)],
            config=types.GenerateContentConfig(temperature=0.2, top_p=0.8)
        )
    return response.text

def analyze_training_data(repo_id: str, episodes: List[int] = None, GPU_POOR=True, rate_limit=15) -> List[Dict]:
    """Analyzes the first frames of each training episode."""
    dataset = LeRobotDataset(repo_id, episodes=episodes) #Load initial dataset
    num_episodes = dataset.num_episodes
    all_episodes_analysis = []
    
    rate_limit = rate_limit if GPU_POOR else None  #  requests per minute
    if rate_limit:
        calls = deque(maxlen=rate_limit)

    for ep_idx in tqdm(range(num_episodes), desc="Analyzing Training Episodes"):
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        #to_idx = dataset.episode_data_index["to"][ep_idx].item()
        
        top_frame = dataset[from_idx]["observation.images.phone"]
        front_frame = dataset[from_idx]["observation.images.laptop"]

        top_image = tensor_to_pil(top_frame)
        front_image = tensor_to_pil(front_frame)

        if rate_limit:
            now = time.time()
            while calls and now - calls[0] > 60:
                calls.popleft()
            if len(calls) >= rate_limit:
                time.sleep(max(0, 60.1 - (now- calls[0])))
        
        top_analysis_raw = analyze_scene(top_image)
        
        if rate_limit:
            calls.append(time.time())
            now = time.time()
            while calls and now - calls[0] > 60:
                calls.popleft()
            if len(calls) >= rate_limit:
                time.sleep(max(0, 60.1 - (now - calls[0])))

        analysis = analyze_multi_view(top_image, front_image, context=top_analysis_raw)
        if rate_limit:
            calls.append(time.time())
        
        analysis = parse_json(analysis)

        all_episodes_analysis.append({"episode": ep_idx, "analysis": analysis})
        
    return all_episodes_analysis

def summarize_training_data(analysis_data: List[Dict]) -> str:
    """Generates a summary of the training data analysis."""
    summary_prompt_template = """
    Analyze the following dataset of robot pick-and-place episodes with lego bricks and containers.
    Summarize patterns, biases, and limitations in the training data.
        
    Focus on:
    1. Object distributions (positions, orientations, colors)
    2. Success patterns
    3. Potential biases (e.g., container always on left)
    4. Limitations in the dataset diversity

    {0}

    Provide the following:
    1.  **Training Statistics:**  Quantify key aspects, such as the percentage of episodes
        where the container is on the left vs. right side of the robot.
    2.  **Potential Biases:** Identify any biases in the data (e.g., only one container color).
    3.  **Suggestions for Data Augmentation:**  Suggest specific augmentations to address
        biases and improve generalization. Categorize suggestions like this:
        - flip_frame:  (If the data is biased towards one side)
        - change_color: object-container:color-yellow  (If container color variety is needed)
        - inpaint_distraction: List_of_distraction_objects:[object1, object2] (If distractions should be removed)
    """
    return get_summary(analysis_data, summary_prompt_template)

def evaluate_episode_success(
        #first_frame: Union[Image.Image, torch.Tensor],
        last_frame: Union[Image.Image, torch.Tensor]
    ) -> Dict:
        """
        Evaluate whether an episode was successful by comparing first and last frames.
        
        Args:
            first_frame: First frame of the episode
            last_frame: Last frame of the episode
            
        Returns:
            Dictionary with success evaluation and reasoning
        """
        prompt = """
        Analyze the final state of this robotics task (picking and placing Lego bricks).
        1.  Is the Lego brick inside the container? (Answer with YES or NO).
        2.  If NO, provide a concise reason for the failure, relating it to the scene.
        Output should be JSON in format: {"success": bool, "failure_reason": str}
        """
        
        # Convert tensors to PIL if needed
        #if isinstance(first_frame, torch.Tensor):
        #    first_frame = tensor_to_pil(first_frame)
        if isinstance(last_frame, torch.Tensor):
            last_frame = tensor_to_pil(last_frame)
            
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[last_frame, prompt],
                config=types.GenerateContentConfig(temperature=0.2)
            )
            
            return response.text
            
        
        except Exception as e:
            print(f"Error evaluating episode success: {e}")
            return {"error": str(e), "success": False, "confidence": 0}
        
def analyze_eval_data(repo_id: str, episodes: List[int] = None, GPU_POOR=True, rate_limit=15) -> List[Dict]:
    """Analyzes the first frames of each training episode."""
    dataset = LeRobotDataset(repo_id, episodes=episodes) #Load initial dataset
    num_episodes = dataset.num_episodes
    all_episodes_analysis = []

    rate_limit = rate_limit if GPU_POOR else None  #  requests per minute
    if rate_limit:
        calls = deque(maxlen=rate_limit)

    for ep_idx in tqdm(range(num_episodes), desc="Analyzing Eval Episodes"):
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        
        top_first = dataset[from_idx]["observation.images.phone"]
        front_first = dataset[from_idx]["observation.images.laptop"]
        top_image = tensor_to_pil(top_first)
        front_image = tensor_to_pil(front_first)

        if rate_limit:
            now = time.time()
            while calls and now - calls[0] > 60:
                calls.popleft()
            if len(calls) >= rate_limit:
                time.sleep(max(0, 60.1 - (now- calls[0])))
        top_analysis_raw = analyze_scene(top_image)
        
        if rate_limit:
            calls.append(time.time())
            now = time.time()
            while calls and now - calls[0] > 60:
                calls.popleft()
            if len(calls) >= rate_limit:
                time.sleep(max(0, 60.1 - (now - calls[0])))
        scene_analysis_raw = analyze_multi_view(top_image, front_image, context=top_analysis_raw)
        scene_analysis = parse_json(scene_analysis_raw)
        
        # TODO: Add front view to the eval analysis
        top_last = dataset[to_idx-1]["observation.images.phone"]
        
        if rate_limit:
            calls.append(time.time())
            now = time.time()
            while calls and now - calls[0] > 60:
                calls.popleft()
            if len(calls) >= rate_limit:
                time.sleep(max(0, 60.1 - (now - calls[0])))
        eval_analysis_raw = evaluate_episode_success(top_last)
        eval_analysis = parse_json(eval_analysis_raw)
        
        if rate_limit:
            calls.append(time.time())

        all_episodes_analysis.append(
            {"episode": ep_idx, "episode_eval": eval_analysis, "scene_analysis": scene_analysis}
        )
        
    return all_episodes_analysis

def get_augmentations(
    evaluation_results: List[Dict], training_summary: str
) -> str:
    """Summarizes evaluation results and suggests augmentations."""

    # Combine evaluation results into a string format
    combined_eval_results = ""
    for result in evaluation_results:
        combined_eval_results += f"Episode {result['episode']}: \n"
        combined_eval_results += f"Scene Description: {result['scene_analysis']} \n"
        combined_eval_results += f"Episode Evaluation: {result['episode_eval']} \n"

    prompt = """
    Based on the training data summary and failed evaluation episodes,
    suggest data augmentations to improve the robot's policy.
        
    Consider the following types of augmentations:
    1. flip_frame: If the training data shows position bias (e.g., container always on left)
    2. change_color: If the training data shows color bias (e.g., container always blue)
    3. inpaint_distraction: If new distractions in scenes affect performance. New distractions are objects that are not the target object and are not part of training data.
    Only include augmentations that are recommended based on the evaluation results. Format the response as JSON with the following structure:
        {
            "recommended_augmentations": {
                    'flip_frame': True, 
                    'change_color': {
                        'object': 'blue container', 
                        'target_color': 'blue'
                    },
                    'inpaint_distraction': {
                        'objects': ['object1', 'object2']
                    }
            },
            "reasoning": "overall explanation of recommendations",
            "expected_improvements": "how these changes should help"
        }
    """
    prompt += f"Here is a summary of the training data used for the initial policy: {training_summary}"
    prompt += f"Here are the evaluation results: {combined_eval_results}"
    response = client.models.generate_content(
        model=PRO_MODEL_ID,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.2)
    )
    
    return response.text
    