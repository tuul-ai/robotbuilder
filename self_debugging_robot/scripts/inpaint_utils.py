import modal
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

from rembg import remove
from google import genai
from google.genai import types
import base64
from io import BytesIO
from .image_utils import tensor_to_pil, reorder_tensor_dimensions
from .gemini_utils import parse_json

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.0-flash"  # Use Gemini 2.0 Flash for 3D capabilities
PRO_MODEL_ID ='gemini-2.0-pro-exp-02-05'

def get_object_mask(gsam, im, object):
    msk = gsam.remote(im, object)
    msk_np = np.array(msk['results'][-2][msk['results'][3].index(object)][0])
    return Image.fromarray(msk_np)

def get_top_empty_space(img) -> str:
    prompt = """Identify at least 1 and no more than 3 regions of empty space on the table surface in this top-down image.
    The regions should be suitable for placing small objects 
    1. without significant overlap with existing objects.
    2. such that when red robot moves to pick and place the lego brick in container the selected empty space should not lay in its trajectory to lego brick or to the container 
     
    Provide the center point of each region.
    The answer should follow the json format: [{"point": [y, x], "label": "empty_space_1"}, ...].
    The points are in [y, x] format normalized to 0-1000."""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[img, prompt],
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text
    
def get_front_empty_space(img_0, img_1, top_view_points) -> str:
    prompt = """For the following images, predict if the points referenced in the first image are in frame.
    If they are, also predict their 2D coordinates.
    Each entry in the response should be a single line and have the following keys:
    If the point is out of frame: 'in_frame': false, 'label' : <label>.
    If the point is in frame: 'in_frame': true, 'point': [y, x], 'label': <label>.
    The points are in [y, x] format normalized to 0-1000. Use the same labels provided in the context."""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[img_0, prompt, top_view_points, img_1],
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text

def gemini_inpaint_image(contents):
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_modalities=['Text', 'Image']
        )
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_data = base64.b64decode(part.inline_data.data)
            edited_image = Image.open(BytesIO(image_data))
            return edited_image
    return None

def paste_topimg_for_gemini_inpaint(source_image, image_patch, top_coordinates):
    """
    Pastes image_patch onto source_image, centered at the given (x, y) coordinates.

    Args:
        source_image: The base PIL Image.
        image_patch: The PIL Image to paste.
        top_coordinates: The top_coordinates of the center of where to paste.
    """
    x = (top_coordinates[1]/1000.0) * source_image.size[0]
    y = (top_coordinates[0]/1000.0) * source_image.size[1]

    width, height = image_patch.size
    top_left_x = int(x - width / 2)
    top_left_y = int(y - height / 2)
    # Create a copy to avoid modifying the original image
    new_image = source_image.copy()
    new_image.paste(image_patch, (top_left_x, top_left_y))
    return new_image

def get_inpaint_object(gsam, obs, object):
    top_view = tensor_to_pil(obs['observation.images.phone'])
    front_view = tensor_to_pil(obs['observation.images.laptop'])

    empty_space_top_raw = get_top_empty_space(top_view)
    empty_space_top = parse_json(empty_space_top_raw)
    
    empty_space_front_raw = get_front_empty_space(top_view, front_view, empty_space_top_raw)
    empty_space_front = parse_json(empty_space_front_raw)

    in_frame_points = [p for p in empty_space_front if p.get('in_frame')]
    front_coordinates = in_frame_points[0]['point']
    for p in empty_space_top:  # Find matching label
        if p['label'] == in_frame_points[0]['label']:
            top_coordinates = p['point']
            break

    object_prompt_front = f"""This is front image of a robot. 
    Can you add a {object} to location y={front_coordinates[0]}, x={front_coordinates[1]}.
    The points [y, x] are in format normalized to 0-1000."""
    contents=[object_prompt_front, front_view]
    first_front_im = gemini_inpaint_image(contents)
    front_im_msk = get_object_mask(gsam, first_front_im, object)

    front_im_msk_rgb = front_im_msk.convert('RGB')
    croped_front = extract_masked_region(first_front_im, front_im_msk_rgb)
    top_wrap = create_top_view(croped_front)
    top_pasted = paste_topimg_for_gemini_inpaint(top_view, top_wrap, top_coordinates)
    
    object_prompt_top = f"""This is top view of a robot. Can you align the {object} and make it realistic in the image and cleanup the distorion around it at location y={top_coordinates[0]}, x={top_coordinates[1]}. 
    DO NOT change or edit the Red Robot Arm, Blue Container or lego brick. The points [y, x] are in format normalized to 0-1000."""
    contents=[object_prompt_top, top_pasted]
    first_top_im = gemini_inpaint_image(contents)
    
    top_im_msk = get_object_mask(gsam, first_top_im, object)

    return first_top_im, first_front_im, top_im_msk, front_im_msk

def create_inpainted_frame(obs, source_top_im, source_front_im, top_im_msk, front_im_msk):
    top_view = tensor_to_pil(obs['observation.images.phone'])
    front_view = tensor_to_pil(obs['observation.images.laptop'])

    top_view_obj = paste_masked_region(source_top_im, top_view, top_im_msk)
    top_msk = rembg_mask(top_view)
    top_im = paste_masked_region(top_view, top_view_obj, top_msk)
    # Get rid of alpha channel
    top_im = top_im.convert('RGB')
    top_np = np.array(top_im)

    front_view_obj = paste_masked_region(source_front_im, front_view, front_im_msk)
    front_msk = rembg_mask(front_view)
    front_im = paste_masked_region(front_view, front_view_obj, front_msk)
    front_im = front_im.convert('RGB')
    front_np = np.array(front_im)

    obs['observation.images.phone'] = reorder_tensor_dimensions(torch.from_numpy(top_np))
    obs['observation.images.laptop'] = reorder_tensor_dimensions(torch.from_numpy(front_np))
    return obs

def create_inpaint_mask(
    im: Union[Image.Image, np.ndarray],
    direction: str,
) -> tuple[Image.Image, Image.Image]:
    """
    Create a mask for inpainting a corner region of the image
    
    Args:
        im: Input image (PIL Image or numpy array)
        direction: Corner to mask - 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    
    Returns:
        tuple: (mask, overlaid_image) where overlaid_image shows the masked region in white
    """
    # Convert numpy array to PIL if needed
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    
    width, height = im.size
    corner_width = width // 4
    corner_height = height // 4
    
    # Create mask (black background with white corner)
    mask = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(mask)
    
    # Define corner coordinates based on direction
    if direction == 'top-left':
        coords = [0, 0, corner_width, corner_height]
    elif direction == 'top-right':
        coords = [width - corner_width, 0, width, corner_height]
    elif direction == 'bottom-left':
        coords = [0, height - corner_height, corner_width, height]
    elif direction == 'bottom-right':
        coords = [width - corner_width, height - corner_height, width, height]
    else:
        raise ValueError("Direction must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")
    
    # Draw white rectangle for the corner in mask
    draw.rectangle(coords, fill='white')
    
    # Create overlaid image by copying original and drawing white rectangle
    overlaid_im = im.copy()
    draw = ImageDraw.Draw(overlaid_im)
    draw.rectangle(coords, fill='white')
    
    return mask, overlaid_im

def create_outpaint_mask(
    im: Image.Image,
    direction: str,
    plot: bool = True
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """
    Create mask for outpainting by extending image in specified direction
    
    Args:
        im: Input PIL Image (keeps original dimensions)
        direction: One of ['left-top', 'left', 'top', 'right-top', 
                         'right', 'right-bottom', 'bottom', 'left-bottom']
        plot: Whether to display visualization
    
    Returns:
        tuple: (modified_image, mask, overlay)
        - modified_image: Original image cropped and extended with white space
        - mask: White in new area, black in original area
        - overlay: Visualization with translucent overlay
    """
    # Get original dimensions
    width, height = im.size
    
    # Calculate crop/extend sizes (1/10 of image)
    shift_x = width // 10
    shift_y = height // 10
    
    # Create new white image with original dimensions
    new_im = Image.new('RGB', (width, height), 'white')
    
    # Create base mask (black)
    mask = Image.new('RGB', (width, height), 'black')
    
    # Process based on direction
    if 'left' in direction:
        paste_x = shift_x
    elif 'right' in direction:
        paste_x = -shift_x
    else:
        paste_x = 0
        
    if 'top' in direction:
        paste_y = shift_y
    elif 'bottom' in direction:
        paste_y = -shift_y
    else:
        paste_y = 0
    
    # Crop original image
    crop_box = (
        max(-paste_x, 0),
        max(-paste_y, 0),
        width - max(paste_x, 0),
        height - max(paste_y, 0)
    )
    cropped_im = im.crop(crop_box)
    
    # Paste cropped image into new white image
    paste_box = (
        max(paste_x, 0),
        max(paste_y, 0),
        width - max(-paste_x, 0),
        height - max(-paste_y, 0)
    )
    new_im.paste(cropped_im, paste_box)
    
    # Create mask (white in new area)
    white_box = []
    if 'left' in direction:
        white_box.append((0, 0, shift_x, height))
    if 'right' in direction:
        white_box.append((width-shift_x, 0, width, height))
    if 'top' in direction:
        white_box.append((0, 0, width, shift_y))
    if 'bottom' in direction:
        white_box.append((0, height-shift_y, width, shift_y))
    
    # Fill mask with white in new areas
    mask_draw = ImageDraw.Draw(mask)
    for box in white_box:
        mask_draw.rectangle(box, fill='white')
    
    return new_im, mask

def create_top_view(
    im: Union[Image.Image, np.ndarray],
    rotation_angle: float = 180.0,
    perspective_strength: float = 0.3
) -> Image.Image:
    """
    Rotate image and apply perspective transform to simulate top view
    
    Args:
        im: Input image (PIL Image or numpy array)
        rotation_angle: Rotation angle in degrees
        perspective_strength: Strength of perspective transform (0 to 1)
    
    Returns:
        Image.Image: Transformed image
    """
    # Convert numpy array to PIL if needed
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    
    # Get image dimensions
    width, height = im.size
    
    # First rotate the image
    rotated_im = im.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
    
    # Calculate perspective transform points
    # Source points (corners of original image)
    src_points = np.float32([
        [0, 0],  # top-left
        [width, 0],  # top-right
        [width, height],  # bottom-right
        [0, height]  # bottom-left
    ])
    
    # Calculate destination points for perspective transform
    perspective_shift = int(height * perspective_strength)
    dst_points = np.float32([
        [perspective_shift, perspective_shift],  # top-left moved down and right
        [width - perspective_shift, perspective_shift],  # top-right moved down and left
        [width, height],  # bottom-right stays
        [0, height]  # bottom-left stays
    ])
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective transform
    rotated_arr = np.array(rotated_im)
    result_arr = cv2.warpPerspective(
        rotated_arr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR
    )
    
    # Convert back to PIL Image
    result_im = Image.fromarray(result_arr)
    
    return result_im

def extract_masked_region(
    im: Union[Image.Image, np.ndarray],
    mask: Image.Image
) -> Image.Image:
    """
    Extract/crop the region from image where mask is white
    
    Args:
        im: Input image (PIL Image or numpy array)
        mask: Mask image (white pixels indicate region to extract)
    
    Returns:
        Image.Image: Cropped region from original image
    """
    # Convert numpy array to PIL if needed
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    
    # Convert mask to numpy array
    mask_arr = np.array(mask)
    
    # Find white pixels (assuming RGB mask where white is [255,255,255])
    white_pixels = np.all(mask_arr == 255, axis=2)
    white_coords = np.where(white_pixels)
    
    # Get bounding box of white region
    min_y, max_y = white_coords[0].min(), white_coords[0].max()
    min_x, max_x = white_coords[1].min(), white_coords[1].max()
    
    # Crop original image to this region
    cropped_region = im.crop((min_x, min_y, max_x + 1, max_y + 1))
    
    return cropped_region

def paste_wrapped_image(
    target_im: Image.Image,
    wrapped_im: Image.Image,
    mask: Image.Image,
    direction: str
) -> tuple[Image.Image, Image.Image]:
    """
    Paste wrapped trapezoid image into target image's masked region
    
    Args:
        target_im: Target image with masked region
        wrapped_im: Perspective-transformed image (with black background)
        mask: Original mask (white region indicates where to paste)
        direction: One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    
    Returns:
        tuple: (modified_target, new_mask)
    """
    # Convert to PIL if needed
    if isinstance(target_im, np.ndarray):
        target_im = Image.fromarray(target_im)
    if isinstance(wrapped_im, np.ndarray):
        wrapped_im = Image.fromarray(wrapped_im)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    
    # Extract non-black regions from wrapped image
    wrapped_arr = np.array(wrapped_im)
    non_black = np.any(wrapped_arr > 10, axis=2)  # Threshold to handle compression artifacts
    coords = np.where(non_black)
    min_y, max_y = coords[0].min(), coords[0].max()
    min_x, max_x = coords[1].min(), coords[1].max()
    
    # Create RGBA version of wrapped image
    wrapped_rgba = Image.new('RGBA', wrapped_im.size, (0,0,0,0))
    wrapped_rgba.paste(wrapped_im)
    wrapped_data = np.array(wrapped_rgba)
    wrapped_data[~non_black] = [0,0,0,0]  # Make black pixels transparent
    trapezoid = Image.fromarray(wrapped_data)
    
    # Find mask coordinates (white region)
    mask_arr = np.array(mask)
    white_pixels = np.all(mask_arr == 255, axis=2)
    mask_coords = np.where(white_pixels)
    mask_min_y, mask_max_y = mask_coords[0].min(), mask_coords[0].max()
    mask_min_x, mask_max_x = mask_coords[1].min(), mask_coords[1].max()
    
    # Create result image with white masked region
    result_im = target_im.copy()
    draw = ImageDraw.Draw(result_im)
    draw.rectangle([mask_min_x, mask_min_y, mask_max_x, mask_max_y], fill='white')
    
    # Calculate paste coordinates based on direction
    if 'top' in direction:
        paste_y = mask_min_y  # Align to top
        if 'left' in direction:
            paste_x = mask_min_x  # Align to left
        else:  # top-right
            paste_x = mask_max_x - (max_x - min_x)  # Align to right
    else:  # bottom
        paste_y = mask_max_y - (max_y - min_y)  # Align to bottom
        if 'left' in direction:
            paste_x = mask_min_x  # Align to left
        else:  # bottom-right
            paste_x = mask_max_x - (max_x - min_x)  # Align to right
    
    # Paste trapezoid image
    result_im.paste(trapezoid, (paste_x - min_x, paste_y - min_y), trapezoid)
    
     # Create new mask for the remaining white space
    new_mask = Image.new('RGB', target_im.size, 'black')
    draw = ImageDraw.Draw(new_mask)
    
    # First fill the entire original masked region
    draw.rectangle([mask_min_x, mask_min_y, mask_max_x, mask_max_y], fill='white')
    
    # Create mask array
    mask_arr = np.array(new_mask)
    
    # Get the region where the trapezoid was pasted
    paste_height = max_y - min_y + 1
    paste_width = max_x - min_x + 1
    
    # First make the entire pasted region black in the mask
    paste_region = np.zeros_like(mask_arr[:,:,0], dtype=bool)
    paste_region[paste_y:paste_y+paste_height, paste_x:paste_x+paste_width] = True
    mask_arr[paste_region] = [0,0,0]
    
    # Then make the black pixels from wrapped image white in the mask
    black_pixels = ~np.any(wrapped_arr > 10, axis=2)  # Find black pixels
    black_region = black_pixels[min_y:max_y+1, min_x:max_x+1]  # Get region within bounds
    
    # Apply black pixels as white to the mask
    paste_mask = np.zeros_like(mask_arr[:,:,0], dtype=bool)
    paste_mask[paste_y:paste_y+paste_height, paste_x:paste_x+paste_width] = black_region
    mask_arr[paste_mask] = [255,255,255]
    
    new_mask = Image.fromarray(mask_arr)
    
    return result_im, new_mask

def paste_masked_region_old(
    target_im: Image.Image,
    original_im: Image.Image, 
    mask: Image.Image
) -> Image.Image:
    """
    Paste content from original image into target image where mask is white
    
    Args:
        target_im: Target image to paste into
        original_im: Source image to copy content from
        mask: Mask where white pixels indicate where to paste
    
    Returns:
        Image.Image: Modified target image
    """
    # Convert to PIL if needed
    if isinstance(target_im, np.ndarray):
        target_im = Image.fromarray(target_im)
    if isinstance(original_im, np.ndarray):
        original_im = Image.fromarray(original_im)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
        
    # Create result image
    result_im = target_im.copy()
    
    # Convert mask to RGBA for alpha compositing
    mask_rgba = mask.convert('RGBA')
    mask_data = np.array(mask_rgba)
    
    # Make black regions transparent in mask
    mask_data[np.all(mask_data[:,:,:3] == 0, axis=2)] = [0,0,0,0]
    mask_data[np.all(mask_data[:,:,:3] == 255, axis=2)] = [255,255,255,255]
    alpha_mask = Image.fromarray(mask_data)
    
    # Create RGBA version of original image
    original_rgba = Image.new('RGBA', original_im.size)
    original_rgba.paste(original_im)
    
    # Paste original content using mask as alpha
    result_im.paste(original_rgba, (0,0), alpha_mask)
    
    return result_im

def paste_masked_region(source_img: Image.Image, target_img: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Extract the masked region from image_0 and paste it on image_1 using given mask.
    
    Args:
        image_0: Source image.
        image_1: Target image.
        mask: Mask image (grayscale, white indicates region to extract).
        
    Returns:
        Image.Image: image_1 with masked region from image_0 pasted.
    """
    
    # Ensure images are in RGBA format to handle transparency
    source_img = source_img.convert("RGBA").resize(target_img.size)
    target_img = target_img.convert("RGBA")
    mask = mask.convert("L").resize(target_img.size)  # Ensure mask is grayscale

    # Create a copy of image_1 to modify
    new_target_img = target_img.copy()

    # Extract the masked region from image_0
    masked_region = Image.composite(source_img, Image.new('RGBA', source_img.size, (0,0,0,0)), mask)

    # Paste the masked region onto image_1
    new_target_img.paste(masked_region, (0, 0), mask)

    return new_target_img

def rembg_mask(im: Image.Image) -> Image.Image:
    obj = remove(im)
    obj_msk = Image.new("L", im.size)
    obj_msk.paste(obj, (0, 0), obj)

    msk = obj_msk.point(lambda x: 0 if x else 255, "1").convert('L')
    negative_msk = msk.point(lambda x: 255 - x, "1").convert('L')
    return negative_msk