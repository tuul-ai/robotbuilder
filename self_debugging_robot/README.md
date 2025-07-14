# Robotics Experiment Log: Foundation Models, Synthetic Data & Robot Character

> This repository serves as a log for my ongoing experiments exploring the intersection of robotics foundation models, synthetic data generation, and imbuing robots with character traits like context awareness, memory, and agentic behavior. Here, you'll find experimental notebooks, results, code snippets, and eventually, videos showcasing these explorations.

The initial focus is on building a self-improving robotics system, nicknamed "Caferacer," where a robot can analyze its own performance, identify weaknesses, and automatically generate targeted training data to improve.

## Core Idea (Initial Experiment: Self-Debugging Robot)

The first set of experiments documented here focuses on enabling robots to move beyond static datasets by:

1.  **Analyzing their own performance**: Assessing success/failure and the *why* behind it.
2.  **Identifying specific weaknesses**: Pinpointing issues like positional biases or color dependencies.
3.  **Generating targeted synthetic data**: Creating new training examples to address identified flaws.
4.  **Retraining and iterating**: Continuously improving the robot's policy through this closed loop.

## Current Experiments & Tools

This section details the tools and experiments already logged in this repository, primarily focusing on the initial "self-debugging" concept.

### 1. Gemini-Powered Analysis Engine

Utilizes Google's Gemini models for analysis:

-   **Gemini 2.0 Flash**: Analyzes visual scenes from multiple camera perspectives.
-   **Gemini 2.0 Pro**: Summarizes training data, evaluates failures, and recommends data augmentation strategies based on observations.

**Key Utility Functions (`gemini_utils.py`):**

-   `analyze_scene`: Analyzes a single frame.
-   `analyze_multi_view`: Combines analysis from multiple camera views.
-   `get_summary`: Generates summaries from analysis results.
-   `analyze_training_data`/`analyze_eval_data`: Processes entire datasets.
-   `get_augmentations`: Recommends specific augmentations based on analysis.

**Example Usage (Analysis & Recommendation):**

```python
# Found in: 00_gemini_loop.ipynb
from scripts.gemini_utils import (
    analyze_training_data, 
    summarize_training_data,
    analyze_eval_data, 
    get_augmentations, 
    parse_json
)

# Analyze training data
train_results = analyze_training_data("your/training_repo_id", episodes=list(range(10)), GPU_POOR=True)
train_summary = summarize_training_data(train_results)

# Analyze evaluation data and get augmentation recommendations
eval_results = analyze_eval_data("your/eval_repo_id", episodes=list(range(1, 10)))
augmentations_raw = get_augmentations(eval_results, train_summary)
augmentations = parse_json(augmentations_raw)
DATA_AUG = augmentations['recommended_augmentations']
print(DATA_AUG) 
```

### 2. Data Augmentation System

Implements various data augmentation techniques driven by the analysis engine:

-   **flip_frame**: Addresses positional biases by flipping images and adjusting action vectors.
-   **change_color**: Uses Grounded-SAM and OpenCV to change object colors (e.g., containers, bricks).
-   **inpaint_distraction**: Uses Gemini to generate novel objects in empty scene spaces to test robustness.

**Key Utility Functions:**

-   **`aug_utils.py`**:
    -   `flip_frame`: Flips frame data and actions.
    -   `get_mask`: Uses Grounded-SAM for object segmentation masks.
    -   `change_object_color`/`apply_color`: Modifies colors in images/tensors.
-   **`inpaint_utils.py`**:
    -   `get_object_mask`, `get_top_empty_space`: Identify target areas for inpainting.
    -   `gemini_inpaint_image`: Uses Gemini to generate an image with an added object.
    -   `create_inpainted_frame`: Integrates the inpainted object into a dataset frame.

**Example Usage (Applying Augmentations):**

```python
# Found in: 01_augment_dataset.ipynb
import modal
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# Assuming other necessary imports like create_dataset are defined in the notebook

# Setup grounded-sam for segmentation (if needed for augmentations)
gsam = modal.Function.lookup("grounded-sam", "GroundedSam.run", environment_name='prod')

# Load base dataset
dataset0 = LeRobotDataset("your/training_repo")

# Define augmentations based on analysis (example)
DATA_AUG = {
    'flip_frame': True, 
    'change_color': {
        'object': 'blue container', 
        'target_color': 'yellow'
    }
}

# Create augmented dataset (using a function defined in the notebook)
# dataset = create_dataset("your/new_repo", dataset0, gsam, DATA_AUG=DATA_AUG) 
# dataset.push_to_hub()
```

### 3. Notebooks & Workflow

The core experimental logic is currently captured in Jupyter notebooks:

-   `00_gemini_loop.ipynb`: Focuses on analyzing training/evaluation data and generating augmentation recommendations using the Gemini Brain.
-   `01_augment_dataset.ipynb`: Implements the recommended augmentations using the Data Augmentation System to create new datasets.

*Note: For GPU-limited environments, use the `GPU_POOR=True` flag in analysis functions to optimize API calls.*

## Planned Experiments & Future Directions

This repository will evolve to include explorations into:

-   **Robot Memory**: Implementing short-term and long-term memory mechanisms.
-   **Context Awareness**: Enabling robots to understand and react to their broader environment and task history.
-   **Agentic Behavior**: Developing systems where robots can set their own sub-goals or adapt strategies based on experience.
-   **Advanced Synthetic Data**: Moving beyond simple augmentations to generating more complex and realistic scenarios.
-   **Foundation Model Integration**: Testing different foundation models for planning, perception, and control.
-   **Video Documentation**: Adding video examples of experiments.

## Getting Started with Current Code

Ensure you have the necessary dependencies installed (refer to environment setup if applicable, e.g., potentially using `conda` or `pip`). The primary dependencies include `lerobot`, `google-generativeai`, `modal-client`, `opencv-python`, `Pillow`, `torch`, `numpy`, etc.

Add a path to the LEROBOT repository in your python path. Clone the repository and explore the Jupyter notebooks (`*.ipynb`) to understand the current experimental flow.

## Limitations (Current Experiments)

The initial self-debugging experiments are focused on the LeRobot dataset and SO100 robot arm:

-   Primarily tested with pick-and-place tasks involving Lego bricks.
-   Requires Gemini API access and potentially Modal for segmentation.
-   Can be GPU-intensive for certain operations (though `GPU_POOR` options exist).

## Collaboration & Contact

If you're interested in collaborating or adapting these ideas, feel free to reach out:

-   Email: hey@tuul.ai
-   Twitter: [@shreyasgite](https://x.com/shreyasgite)

## License

[MIT License](LICENSE)
