import cv2
import numpy as np
from PIL import Image

def gaussian_blur_app(image, sigma, pipe):
    """
    Apply Gaussian blur to the background of an image based on segmentation.
    
    Args:
        image: PIL Image uploaded by the user
        sigma: Blur intensity (sigma value for Gaussian blur)
        pipe: The segmentation pipeline
        
    Returns:
        PIL Image with blurred background
    """
    # Generate the binary mask using the pipeline
    pillow_mask = pipe(image, return_mask=True)
    
    original_image_np = np.array(image)
    mask = np.array(pillow_mask)
    
    # Apply Gaussian blur to entire image
    blurred_image = cv2.GaussianBlur(original_image_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # Convert 2D mask to 3D if necessary
    if len(mask.shape) == 2:
        mask_3d = np.stack([mask] * 3, axis=-1)
    else:
        mask_3d = mask
    
    mask_normalized = mask_3d / 255.0
    
    # Combine foreground and background
    final_image = (mask_normalized * original_image_np + (1 - mask_normalized) * blurred_image).astype(np.uint8)
    
    return Image.fromarray(final_image)