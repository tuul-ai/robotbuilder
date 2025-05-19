# Lesson 2: Deep Dive into Gemini Robotics

## Building an Agentic Robot with Gemini 2.0 Flash + pi0

Build an agentic robot that does your bidding. So you can now verbally tell the robot, "I'm building a red Lego wall or wooden tower", and it will infer the next steps by itself and pass me the necessary pieces, tools, or materials, ha!
You can also just ask it to bring you things!

### Pipeline Overview

The pipeline works as follows:
- OpenAI Whisper (local) → speech to text
- Gemini → makes sense of user requests, converts to robot tasks, bounding boxes, grasping points, etc. (System 2 thinking)
- π0 → robotic actions (System 1)

## Understanding Gemini Robotics

The [Gemini robotics paper](https://arxiv.org/pdf/2503.20020) can be broken down into 3 parts:
- Exploring the Embodied reasoning capabilities of Gemini. [Demo](https://github.com/tuul-ai/robotbuilder/blob/main/notebooks/gemini_spatial_3d.ipynb) capabilities of Gemini
- Gemini out of the box as robotic controller. [Demo](https://x.com/shreyasgite/status/1888109203606188464) 
- Gemini ER + Local action decoder model as robotic foundational model. [Demo](https://x.com/shreyasgite/status/1923008943938244698)

### Comparison with Other Models

One of the main differences between Gemini robotics vs [pi0](https://www.physicalintelligence.company/blog/pi0) & [GrootN1](https://arxiv.org/pdf/2503.14734) is that:
- Gemini Robotics uses two VLAs: one is the finetuned Gemini with Embodied reasoning, and other distilled VLA run locally for action decoder.  
- pi0 and Groot N1 use only one VLA directly connected to the Action decoder.

### How It Works

- Gemini ER (cloud backbone) takes in Images and a general task from the user. And outputs subtask and coordinates (bbox, grasp points, trajectories etc). 
- The local action decoder takes in images & robot state and output from Gemini ER and outputs action sequence.

## Implementation Approach

Since we don't have access to finetuned Gemini ER, we will use:
- Gemini Flash 2.0 (good enough)
- pi0 instead of Gemini robotics decoder

Here is the [video walk through](https://youtu.be/M5YJI3i2ul0).

## Training Data Pipeline

We obviously don't have 10K+ hours of robotics data like Google, but for start let's finetune π0 for pick-and-place few objects (e.g. legos or toys etc), and watch how beautifully it generalizes to all kinds of tasks.

How the training data pipeline works:
- Generate BBoxes for all pick-and-place objects in the scene (Using Gemini)  
- Pick-and-place targets are selected randomly  
- Add the BBox coordinates to the robot's state  
- Overlay the BBoxes in the visualization so you know what to grab and where to drop

## Implementation

It is assumed that you have SO100 or SO101 from lerobot setup and have done the vanilla act training from lesson 1 (lesson_1 is WIP).

1. Update your [`manipulator.py`](https://github.com/huggingface/lerobot/blob/main/lerobot/common/robot_devices/robots/manipulator.py) file: Make necessary changes to robot states on line 188:
   ```python
   def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms) + ["pick_y1", "pick_x1", "pick_y2", "pick_x2"] + ["place_y1", "place_x1", "place_y2", "place_x2"]
   ```
2. Replace your [`control_utils.py`](https://github.com/huggingface/lerobot/blob/main/lerobot/common/robot_devices/control_utils.py) file with [the file from this repo](https://github.com/tuul-ai/robotbuilder/blob/main/notebooks/scripts/control_utils.py).
3. Add `gemini_perception.py` file to a new directory called `vision_utils`, so the new path will look like: `lerobot/common/vision_utils/gemini_perception.py`

Notes: (Working on making this more user friendly and generic, but for now this is the best way to do it)
- If you are not using bins or containers for place update prompt in `get_2d_bbox` in [`gemini_perception.py`](https://github.com/tuul-ai/robotbuilder/blob/main/notebooks/scripts/gemini_perception.py)
- Or pass on the prompt to `get_target_bbox` with `place_locations` in `gemini_perception`
- And update the argument `place_location="bin"` in `create_pick_place_list`
- Add `--control.display_data=true` for visualization while running the [lerobot data recording command](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md#g-record-a-dataset)

## Inference Process

During inference:  
- Generate BBoxes for every object again
- Click the object you want to pick and its target spot; those BBoxes get added to the robot state  
- Let the robot do the work for you 😃

For inference use `robot_inference.ipynb` notebook or make necessary changes to `control_util` files.

## Potential Improvements

Things that could help:
- Conditioning on grasping points
- Better data collection (We need quality teleop data)
- Lots more synthetic data and simulations (part of next modules)

## Setup Instructions

### Adding Gemini API Key

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Setting Lerobot as Working Directory

```bash
cd path/to/lerobot
```

Replace `path/to/lerobot` with the actual path to your lerobot installation.

## Model Training Instructions

We are Modal for compute and training policies. [Jakub](https://www.linkedin.com/in/jakubcieslik/) is working on video tutorial to help you set up your own training.
If there is enough interest, we can make our internal plug-play tool for training and deploying robotics foundational models.

## Contact

If you have any questions or want to point out any issues, please reach out to us at hey@tuul.ai or via [x @shreyasgite](https://x.com/shreyasgite) or [linkedin](https://www.linkedin.com/in/shreyasgite/).
