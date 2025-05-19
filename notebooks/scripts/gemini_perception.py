import os
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import torch
import numpy as np
import json
import time
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.0-flash"  # Use Gemini 2.0 Flash for 3D capabilities
PRO_MODEL_ID ='gemini-2.0-pro-exp-02-05'

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a PyTorch tensor to PIL Image"""
    if len(tensor.shape) == 4:  # Remove batch dimension if present
        tensor = tensor[0]
    
    # Convert from CxHxW to HxWxC
    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        img_array = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        img_array = tensor.cpu().numpy()
    
    # Scale to 0-255 if needed
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)
def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output
    
def get_2D_bbox(img, prompt=None) -> str:
    """Prompts Gemini 2.0 Flash 2D bounding box."""
    bounding_box_system_instructions = """
    You are an expert at analyzing images to identify and locate objects.
    Return bounding boxes as a JSON array. Each object in the array should have a "label" (string) and "box_2d" (array of 4 numbers).
    The "box_2d" coordinates must be [ymin, xmin, ymax, xmax], normalized to a 0-1000 scale.
    Never return Python code fencing (```python ... ```) or general markdown fencing (``` ... ```) around the JSON. Only output the raw JSON array.
    If an object is present multiple times, name them uniquely (e.g., "red lego brick 1", "red lego brick 2") """

    if prompt is None:
        prompt = """Analyze the provided image. Detect all distinct lego bricks, small toys, and any items that could be considered a 'blue bin' or a 'yellow bin' present on the desk.
        Ignore the robot arm itself if visible.
        Return your findings strictly as a JSON array, following the format specified in the system instructions.
        Example of the expected JSON output format: [{"label": "blue lego brick", "box_2d": [100, 200, 150, 280]}, {"label": "yellow bin", "box_2d": [500, 600, 700, 850]}]"""
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[img, prompt],
        config=types.GenerateContentConfig(system_instruction=bounding_box_system_instructions, temperature=0.5),
    )
    return response.text

def plot_bbox(im_tensor, pick_bbox, label=None, place_bbox=None, place_label=None):
    """
    Plots pick and place bounding boxes on an image with different colors.

    Args:
        im_tensor: The image tensor
        pick_bbox: Pick bounding box in normalized [y1, x1, y2, x2] format
        label: Text label for pick object
        place_bbox: Place bounding box in normalized [y1, x1, y2, x2] format
        place_label: Text label for place object
    """
    # Load the image
    im = tensor_to_pil(im_tensor)
    img = im.copy()
    width, height = img.size
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define colors
    pick_color = 'red'
    place_color = 'blue'

    # Helper function to draw a bbox
    def draw_bbox_with_label(bbox, color, label_text=None):
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bbox[0]/1000 * height)
        abs_x1 = int(bbox[1]/1000 * width)
        abs_y2 = int(bbox[2]/1000 * height)
        abs_x2 = int(bbox[3]/1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
        
        # Draw label if provided
        if label_text:
            # Choose font and font size
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("Arial.ttf", 20)
            except:
                font = None
                
            # Add text background for better visibility
            text_bbox = draw.textbbox((abs_x1, abs_y1 - 25), label_text, font=font)
            draw.rectangle(text_bbox, fill='black')
            draw.text((abs_x1, abs_y1 - 25), label_text, fill='white', font=font)
    
    # Draw pick bbox
    if pick_bbox is not None and not (pick_bbox == [0, 0, 0, 0]):
        draw_bbox_with_label(pick_bbox, pick_color, label)
    
    # Draw place bbox if provided
    if place_bbox is not None:
        draw_bbox_with_label(place_bbox, place_color, place_label)

    return img

def create_pick_place_lists(objects_list, place_location="bin"):
    # Make sure we're working with a list of dictionaries
    # TODO: Add support for multiple place locations
    pick = []
    place = []
    pick_labels = []
    place_labels = []
    
    for obj in objects_list:
        if place_location in obj["label"].lower():
            place.append(obj["box_2d"])
            place_labels.append(obj["label"])
        else:
            pick.append(obj["box_2d"])
            pick_labels.append(obj["label"])
    
    print(f"Found {len(place)} place objects and {len(pick)} pick objects")
    return pick, place, pick_labels, place_labels

def normalize_bbox_0to1(bbox):
    normalized_bbox = []
    for p in bbox:
        normalized_bbox.append([p[0]/1000, p[1]/1000, p[2]/1000, p[3]/1000])
    return normalized_bbox

def get_target_bbox(img_tensor, prompt=None):
    img = tensor_to_pil(img_tensor)
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            bbox = get_2D_bbox(img, prompt=prompt)
            bounding_boxes = parse_json(bbox)
            
            if isinstance(bounding_boxes, str):
                bounding_boxes = json.loads(bounding_boxes)
                
            pick, place, pick_labels, place_labels = create_pick_place_lists(bounding_boxes)
            
            # Check if we found at least one pick and one place object
            if len(place) > 0:  # We need at least a place object
                normalized_pick = normalize_bbox_0to1(pick)
                normalized_place = normalize_bbox_0to1(place)
                return pick, place, normalized_pick, normalized_place, pick_labels, place_labels
                
            print(f"Attempt {attempt+1}: No place objects found, retrying...")
                
        except json.JSONDecodeError:
            print(f"Attempt {attempt+1}: JSON parsing error, retrying...")
            import time
            time.sleep(1)  # Small delay before retry
    
    # Fallback if all attempts fail
    print("All attempts failed - using default values")
    default_pick = [[100, 200, 150, 250]]
    default_place = [[300, 400, 350, 450]]
    default_pick_labels = ["default_object"]
    default_place_labels = ["default_bin"]
    return default_pick, default_place, normalize_bbox_0to1(default_pick), normalize_bbox_0to1(default_place), default_pick_labels, default_place_labels

def get_random_targets(pick, place, normalized_pick, normalized_place, pick_labels=None, place_labels=None):
    if len(pick) == 0:
        pick_target = [0, 0, 0, 0]
        place_target = place[0]
        normalized_pick_target = pick_target
        normalized_place_target = normalized_place[0]
        pick_target_label = "none"
        place_target_label = place_labels[0] if place_labels else "bin"
    else:
        pick_idx = np.random.randint(0, len(pick))
        place_idx = np.random.randint(0, len(place))
        pick_target = pick[pick_idx]
        place_target = place[place_idx]
        normalized_pick_target = normalized_pick[pick_idx]
        normalized_place_target = normalized_place[place_idx]
        pick_target_label = pick_labels[pick_idx] if pick_labels else "object"
        place_target_label = place_labels[place_idx] if place_labels else "bin"
        
        # Create new lists with the selected pick target removed (all using the same index)
        pick = [p for i, p in enumerate(pick) if i != pick_idx]
        normalized_pick = [p for i, p in enumerate(normalized_pick) if i != pick_idx]
        
        if pick_labels:
            pick_labels = [p for i, p in enumerate(pick_labels) if i != pick_idx]
            
    return pick_target, place_target, torch.tensor(normalized_pick_target), torch.tensor(normalized_place_target), pick, normalized_pick, pick_target_label, place_target_label, pick_labels
