import torch
from PIL import Image
import numpy as np
import inspect
import matplotlib.pyplot as plt
from tqdm import tqdm

def reorder_tensor_dimensions(tensor):
    """
    Reorders tensor dimensions from (C, H, W) to (H, W, C) format if necessary.
    
    Args:
        tensor: Input tensor with shape (C, H, W) or (H, W, C)
        
    Returns:
        Tensor with shape (H, W, C)
    """
    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        # Permute dimensions from (C, H, W) to (H, W, C)
        tensor = tensor.permute(1, 2, 0)
    else:
        assert tensor.shape[2] == 3, "Invalid tensor shape"
    return tensor

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

# Add to a new cell
def prepare_image_for_gemini(tensor_image):
    """Convert tensor image to PIL format for Gemini Vision API"""
    # Handle different tensor formats
    if len(tensor_image.shape) == 4:  # If it has a batch dimension
        tensor_image = tensor_image[0]
        
    # Convert to numpy and ensure proper range
    img_np = tensor_image.permute(1, 2, 0).cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to PIL
    pil_image = Image.fromarray(img_np)
    return pil_image

def extract_key_frames(dataset):
    """
    Extract first, middle, and last frames from each episode in the dataset.
    
    Args:
        dataset: LeRobotDataset instance
    
    Returns:
        dict: Dictionary with episode indices as keys and frame data as values
    """
    episode_frames = {}
    
    for ep_idx in tqdm(range(dataset.num_episodes), desc="Extracting key frames"):
        # Get frame indices for this episode
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        
        # Calculate middle frame index
        #mid_idx = from_idx + (to_idx - from_idx) // 2
        mid_idx = from_idx + 250
        
        # Extract frames
        first_frame = dataset[from_idx]
        middle_frame = dataset[mid_idx]
        last_frame = dataset[to_idx - 1]  # -1 because to_idx is exclusive
        
        # Extract just the images we need
        episode_frames[ep_idx] = {
            "first": {
                "observation.images.phone": first_frame['observation.images.phone'].clone(),
                "observation.images.laptop": first_frame['observation.images.laptop'].clone(),
            },
            "middle": {
                "observation.images.phone": middle_frame['observation.images.phone'].clone(),
                "observation.images.laptop": middle_frame['observation.images.laptop'].clone(),
            },
            "last": {
                "observation.images.phone": last_frame['observation.images.phone'].clone(),
                "observation.images.laptop": last_frame['observation.images.laptop'].clone(),
            },
            "metadata": {
                "from_idx": from_idx,
                "to_idx": to_idx,
                "length": to_idx - from_idx
            }
        }
    
    return episode_frames

def display_images(*images, titles=None, figsize=(15, 10), max_cols=4):
    """
    Display multiple images in a grid layout with automatic titles based on variable names.
    
    Args:
        *images: Variable number of images (2-8) as PIL Images, numpy arrays, or torch tensors
        titles: Optional list of custom titles. If None, variable names will be used
        figsize: Figure size as (width, height) tuple
        max_cols: Maximum number of columns in the grid
    
    Example:
        phone_im = tensor_to_pil(frame_data['observation.images.phone'])
        laptop_im = tensor_to_pil(frame_data['observation.images.laptop'])
        display_images(phone_im, laptop_im)  # Will use "phone_im" and "laptop_im" as titles
    """
    # Get variable names from the calling frame
    if titles is None:
        frame = inspect.currentframe().f_back
        calling_vars = frame.f_locals.items()
        titles = []
        
        # Find variable names by comparing their values with our images
        for img in images:
            found = False
            for var_name, var_val in calling_vars:
                if var_val is img:
                    titles.append(var_name)
                    found = True
                    break
            if not found:
                titles.append(f"Image {len(titles)+1}")
    
    # Validate number of images
    num_images = len(images)
    if num_images < 2 or num_images > 8:
        raise ValueError("Number of images must be between 2 and 8")
    
    # Determine grid layout
    cols = min(num_images, max_cols)
    rows = (num_images + cols - 1) // cols
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Display each image
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        
        # Convert image to displayable format if needed
        if isinstance(img, torch.Tensor):
            # Convert torch tensor to numpy array
            img_np = img.cpu().numpy()
            # Handle different tensor formats
            if len(img_np.shape) == 3:
                if img_np.shape[0] == 3:  # CHW format
                    img_np = np.transpose(img_np, (1, 2, 0))
            plt.imshow(img_np)
        elif isinstance(img, np.ndarray):
            plt.imshow(img)
        elif isinstance(img, Image.Image):
            plt.imshow(np.array(img))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        
        plt.axis('off')
        plt.title(title)
    
    plt.tight_layout()
    plt.show()