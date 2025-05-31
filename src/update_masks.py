import sys
from pathlib import Path

def process_existing_masks(results_dir, dilation_size=5):
    """
    Process all existing masks in the results directory to apply dilation.
    The dilation is applied to the inpainting region (0s) to expand it.
    
    Args:
        results_dir (str or Path): Path to the results directory
        dilation_size (int): Size of the dilation kernel
    """
    import cv2
    import numpy as np
    from PIL import Image
    
    results_dir = Path(results_dir) / "inpainting_tests"
    
    # Create kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                     (dilation_size, dilation_size))
    
    # Process each image directory
    for image_dir in results_dir.glob("image_*"):
        mask_path = image_dir / "mask.png"
        if mask_path.exists():
            print(f"Processing {mask_path}")
            
            # Read mask
            mask = np.array(Image.open(mask_path))
            
            # Convert to binary format (0s and 1s)
            # Note: in the mask, 0 = inpainting region, 255 = keep region
            mask_binary = (mask > 127).astype(np.uint8) * 255
            
            # Invert to dilate the 0s
            inverted = 255 - mask_binary
            
            # Dilate the inverted mask (expanding 0s)
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            
            # Invert back
            final_mask = 255 - dilated
            
            # Save the dilated mask
            mask_image = Image.fromarray(final_mask.astype(np.uint8))
            mask_image.save(mask_path)
            print(f"Updated {mask_path}")

def main():
    results_dir = Path("results")
    dilation_size = 5  # Can be adjusted as needed
    
    process_existing_masks(results_dir, dilation_size=dilation_size)

if __name__ == "__main__":
    main() 