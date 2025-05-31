import os
from pathlib import Path
from PIL import Image
import numpy as np

def resize_with_pad(im, desired_size):
    """Resize image maintaining aspect ratio and pad to square"""
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
        
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.Resampling.LANCZOS)
    
    # Center pad to desired size
    new_im = Image.new("RGB" if im.mode == "RGB" else "L", 
                      (desired_size, desired_size), 
                      "white" if im.mode == "L" else (0,0,0))
    new_im.paste(im, ((desired_size-new_size[0])//2, 
                     (desired_size-new_size[1])//2))
    return new_im

def resize_directory(image_dir, size=256):
    """Resize all images in a directory to square"""
    # Resize original image
    orig_path = image_dir / "original.png"
    if orig_path.exists():
        img = Image.open(orig_path)
        resized = resize_with_pad(img, size)
        resized.save(orig_path)
        print(f"Resized {orig_path}")
    
    # Resize mask
    mask_path = image_dir / "mask.png"
    if mask_path.exists():
        mask = Image.open(mask_path)
        resized = resize_with_pad(mask, size)
        resized.save(mask_path)
        print(f"Resized {mask_path}")

def main():
    results_dir = Path("results/inpainting_tests")
    
    # Process all image directories
    for dir_path in results_dir.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith("image_"):
            print(f"\nProcessing directory: {dir_path}")
            resize_directory(dir_path)

if __name__ == "__main__":
    main()