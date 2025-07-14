import torch
from PIL import Image
import numpy as np
from typing import Union, Dict
import cv2
import json
import time
from pathlib import Path

from .image_utils import reorder_tensor_dimensions, tensor_to_pil

def flip_frame(obs, action, image_keys):
    """
    Flip frame data horizontally
    """
    for key in image_keys:
        obs[key] = torch.flip(obs[key], [1])
    obs['observation.state'][0] = -obs['observation.state'][0]
    action['action'][0] = -action['action'][0]
    return obs, action

def get_mask(gsam, obs, image_keys, object=None, max_retries=3):
    masks = {}
    for key in image_keys:
        for attempt in range(max_retries): # Max retires in case of network issues
            try:
                im_tensor = obs[key]
                im = tensor_to_pil(im_tensor)
                msk = gsam.remote(im, object)
                msk_np = np.array(msk['results'][-2][msk['results'][3].index(object)][0])
                masks[key] = msk_np
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to get mask after {max_retries} attempts: {e}")
                    raise
                time.sleep(1)  # Wait before retrying
    return masks

def precompute_masks(dataset0, gsam, image_keys, from_idx, to_idx, object=None):
    # Get first frame
    first_frame = {k: reorder_tensor_dimensions(dataset0[from_idx][k]) for k in image_keys}
    mask_0 = get_mask(gsam, first_frame, image_keys, object)
    
    # Get middle frame (8 seconds)
    mid_idx = from_idx + (8 * 30)
    if mid_idx < to_idx:
        mid_frame = {k: reorder_tensor_dimensions(dataset0[mid_idx][k]) for k in image_keys}
        mask_1 = get_mask(gsam, mid_frame, image_keys, object)
    else:
        mask_1 = mask_0
        
    # Get last frame (10 seconds)
    last_idx = from_idx + (10 * 30)
    if last_idx < to_idx:
        last_frame = {k: reorder_tensor_dimensions(dataset0[last_idx][k]) for k in image_keys}
        mask_2 = get_mask(gsam, last_frame, image_keys, object)
    else:
        mask_2 = mask_1

    return mask_0, mask_1, mask_2

def change_object_color(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    mask: Union[Image.Image, np.ndarray],
    target_color: Union[str, tuple, list],
    color_shift_method: str = 'replace',
    preserve_texture: float = 0.3,
    preserve_lighting: bool = True
) -> Image.Image:
    """
    Change the color of an object in an image using a mask.
    
    Args:
        image: Input image (PIL Image, numpy array, or torch tensor)
        mask: Binary mask where white (255) or True represents the object to recolor
        target_color: Target color as string name ('yellow', 'green', 'red', etc.) 
                     or RGB tuple/list (0-255 values)
        color_shift_method: Method to use for color shifting:
                           'replace' - completely replace the color
                           'hue' - shift only the hue, preserve saturation/value
                           'blend' - blend original with target color
        preserve_texture: Amount of original texture to preserve (0.0-1.0)
                         Only used with 'replace' and 'blend' methods
        preserve_lighting: Whether to preserve the original lighting/shading
    
    Returns:
        Image.Image: Image with object color changed
    """
    # Convert image to numpy array if needed
    if isinstance(image, torch.Tensor):
        # Handle PyTorch tensor
        if len(image.shape) == 4:  # Remove batch dimension if present
            image = image[0]
        
        # Convert from CxHxW to HxWxC
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        
        image = image.cpu().numpy()
        
        # Scale to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert mask to numpy array if needed
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Handle different mask formats
    if mask.dtype == bool:
        # Convert boolean mask to uint8 (0 or 255)
        binary_mask = mask.astype(np.uint8) * 255
    elif mask.ndim == 3:
        # Convert RGB mask to grayscale
        if mask.shape[2] == 3:  # RGB
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        elif mask.shape[2] == 4:  # RGBA
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        
        # Threshold to ensure binary mask
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    else:
        # For grayscale masks, ensure values are 0 or 255
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Create color mapping for string color names
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'gray': (128, 128, 128)
    }
    
    # Convert string color to RGB if needed
    if isinstance(target_color, str):
        target_color = target_color.lower()
        if target_color in color_map:
            target_color = color_map[target_color]
        else:
            raise ValueError(f"Unknown color name: {target_color}")
    
    # Ensure target_color is a tuple
    target_color = tuple(target_color)
    
    # Create a copy of the image to modify
    result = image.copy()
    
    if color_shift_method == 'replace':
        # Create a solid color image of the target color
        solid_color = np.zeros_like(image)
        solid_color[:] = target_color
        
        if preserve_texture > 0:
            # Convert to HSV for better color manipulation
            hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv_solid = cv2.cvtColor(solid_color, cv2.COLOR_RGB2HSV)
            
            # Create a blended HSV image
            if preserve_lighting:
                # Keep value (brightness) from original
                hsv_solid[:,:,2] = hsv_img[:,:,2]
            
            # Blend the hue and saturation
            hsv_blend = hsv_img.copy()
            hsv_blend[:,:,0] = hsv_solid[:,:,0]  # Use target hue
            
            # Blend saturation based on preserve_texture
            hsv_blend[:,:,1] = (1 - preserve_texture) * hsv_solid[:,:,1] + preserve_texture * hsv_img[:,:,1]
            
            # Convert back to RGB
            color_img = cv2.cvtColor(hsv_blend, cv2.COLOR_HSV2RGB)
        else:
            color_img = solid_color
            
        # Apply the color change only to the masked region
        mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB) / 255.0
        result = (color_img * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)
        
    elif color_shift_method == 'hue':
        # Convert to HSV for hue manipulation
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Convert target color to HSV to get target hue
        target_color_arr = np.uint8([[list(target_color)]])
        target_hsv = cv2.cvtColor(target_color_arr, cv2.COLOR_RGB2HSV)
        target_hue = target_hsv[0, 0, 0]
        
        # Create a new HSV image with the target hue
        new_hsv = hsv_img.copy()
        new_hsv[:,:,0] = np.where(binary_mask > 0, target_hue, hsv_img[:,:,0])
        
        # Convert back to RGB
        result = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)
        
    elif color_shift_method == 'blend':
        # Create a solid color image
        solid_color = np.zeros_like(image)
        solid_color[:] = target_color
        
        # Create a mask for blending
        mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB) / 255.0
        
        # Blend the original image with the target color
        blend_factor = 1.0 - preserve_texture
        blended = cv2.addWeighted(image, preserve_texture, solid_color, blend_factor, 0)
        
        # Apply the blended color only to the masked region
        result = (blended * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown color shift method: {color_shift_method}")
    
    # Convert back to PIL Image
    return result #Image.fromarray(result)

def apply_color(obs, masks, image_keys, target_color='yellow'):
    
    # TODO: Make it more general, apply color changes to multiple objects in an image using their masks.
    # TODO: Generate masks on the fly, don't load from disk.
    # TODO: Implement inpainting using Flux etc
    # TODO: Use gemini API to inpaint the image.
    # Apply color change to each image using the selected mask
    for key in image_keys:
        obs_np = change_object_color(obs[key], masks[key], target_color, color_shift_method='replace')
        obs[key] = reorder_tensor_dimensions(torch.from_numpy(obs_np))
    return obs

def load_saved_image_dict(save_dir):
    """
    Load saved image dictionary with nested structure.
    
    Args:
        save_dir: Directory where images were saved
    
    Returns:
        Dictionary with same structure as original mask_dict
    """

    
    save_dir = Path(save_dir)
    
    # Load metadata
    with open(save_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create dictionary to store loaded images
    mask_dict = {}
    
    # Load images based on paths in metadata
    for idx_str, temp_pos_dict in metadata['paths'].items():
        idx = int(idx_str)
        mask_dict[idx] = {}
        
        for temp_pos, camera_dict in temp_pos_dict.items():
            mask_dict[idx][temp_pos] = {}
            
            for camera, rel_path in camera_dict.items():
                full_path = save_dir / rel_path
                mask_dict[idx][temp_pos][camera] = Image.open(full_path)
    
    return mask_dict, metadata

def apply_color_augmentation(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    object_masks: Dict[str, Union[Image.Image, np.ndarray]],
    color_map: Dict[str, Union[str, tuple, list]],
    preserve_texture: float = 0.3
) -> Image.Image:
    """
    Apply color changes to multiple objects in an image using their masks.
    
    Args:
        image: Input image
        object_masks: Dictionary mapping object names to their masks
        color_map: Dictionary mapping object names to target colors
        preserve_texture: Amount of original texture to preserve (0.0-1.0)
    
    Returns:
        Image.Image: Image with objects recolored
    
    Example:
        # Apply to bin and robot arm
        result = apply_color_augmentation(
            phone_im,
            {
                'bin': bin_mask,
                'robot_arm': robot_arm_mask
            },
            {
                'bin': 'yellow',
                'robot_arm': 'green'
            }
        )
    """
    # Convert image to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    
    # Start with a copy of the original image
    result = image.copy()
    
    # Apply color changes for each object
    for obj_name, mask in object_masks.items():
        if obj_name in color_map:
            target_color = color_map[obj_name]
            result = change_object_color(
                result, 
                mask, 
                target_color,
                preserve_texture=preserve_texture
            )
    
    return result