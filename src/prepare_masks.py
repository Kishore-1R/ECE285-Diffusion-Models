# notebooks/prepare_masks.ipynb

import sys
import os
from pathlib import Path

# Add the external/segment_anything directory to Python path
current_dir = Path(__file__).parent.parent  # Go up one level from src/
sam_path = current_dir / "external" / "segment-anything"  # or "segment_anything" depending on your folder name
sys.path.append(str(sam_path))

# Print paths for debugging
print("Current directory:", current_dir)
print("SAM path:", sam_path)
print("Python path:", sys.path)

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import cv2

class MaskSelector:
    def __init__(self, data_dir, results_dir):
        """
        Initialize MaskSelector with data and results directories
        
        Args:
            data_dir (Path): Directory containing input images
            results_dir (Path): Directory to save results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir) / "inpainting_tests"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files (both png and jpg)
        self.image_files = sorted(list(self.data_dir.glob("image_*.[pj][np][g]*")))
        if not self.image_files:
            raise ValueError(f"No image files found in {self.data_dir}")
            
        # Load SAM - update path to use models directory
        sam_checkpoint = current_dir / "models" / "sam_vit_h_4b8939.pth"
        if not sam_checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found at {sam_checkpoint}")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        self.sam.to(device)
        self.predictor = SamPredictor(self.sam)

    def load_image(self, image_path):
        """Load image regardless of format"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
        except Exception as e:
            print(f"PIL failed to load image, trying OpenCV: {e}")
            
        # Fallback to OpenCV
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Failed to load image with both PIL and OpenCV: {e}")

    def save_results(self, image, mask, image_name):
        image_dir = self.results_dir / image_name
        image_dir.mkdir(exist_ok=True)
        
        # Save original image
        Image.fromarray(image).save(image_dir / "original.png")
        
        # Invert the mask (1 - mask) so that:
        # 0 = object to remove/inpaint
        # 1 = keep this part of the image
        inverted_mask = (1 - mask) * 255
        mask_image = Image.fromarray(inverted_mask.astype(np.uint8))
        mask_image.save(image_dir / "mask.png")
        print(f"Saved to {image_dir}")

    def process_single_image(self, image_path):
        """Process a single image interactively"""
        image = self.load_image(image_path)
        self.predictor.set_image(image)
        
        while True:
            # Save current figure for reference
            plt.figure(figsize=(10, 5))
            plt.imshow(image)
            plt.savefig('current_image.png')
            plt.close()
            
            print("\nCurrent image saved as 'current_image.png'")
            print("\nEnter points (x,y) coordinates. Format: 'x,y' or 'x,y,label' (label: 1=positive, 0=negative)")
            print("Commands: 'done' to generate masks, 'clear' to reset points, 'quit' to skip image")
            
            points = []
            labels = []
            
            while True:
                try:
                    inp = input("> ").strip()
                    if inp.lower() == 'done':
                        break
                    elif inp.lower() == 'clear':
                        points = []
                        labels = []
                        print("Points cleared")
                        continue
                    elif inp.lower() == 'quit':
                        return
                    
                    # Parse point input
                    parts = inp.split(',')
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        label = int(parts[2]) if len(parts) > 2 else 1  # default to positive
                        points.append([x, y])
                        labels.append(label)
                        print(f"Added point ({x}, {y}) with label {label}")
                    else:
                        print("Invalid format. Use 'x,y' or 'x,y,label'")
                except ValueError:
                    print("Invalid input format")
            
            if not points:
                print("No points added. Try again or 'quit'")
                continue
            
            # Generate masks
            input_points = np.array(points)
            input_labels = np.array(labels)
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            # Show masks and let user select
            for idx, mask in enumerate(masks):
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.subplot(1, 2, 2)
                plt.imshow(image)
                plt.imshow(mask, alpha=0.6)
                plt.title(f'Mask {idx}')
                plt.savefig(f'mask_{idx}.png')
                plt.close()
            
            print("\nMask previews saved as 'mask_[0-2].png'")
            while True:
                try:
                    selection = input("Select mask (0-2) or 'retry' for new points: ")
                    if selection.lower() == 'retry':
                        break
                    mask_idx = int(selection)
                    if 0 <= mask_idx < len(masks):
                        # Save the results
                        self.save_results(image, masks[mask_idx], image_path.stem)
                        return
                    else:
                        print("Invalid mask index")
                except ValueError:
                    print("Invalid input")

    def process_all_images(self):
        """Process all images in the data directory"""
        for image_path in self.image_files:
            print(f"\nProcessing {image_path.name}")
            self.process_single_image(image_path)
            # Cleanup temporary files
            for f in ['current_image.png'] + [f'mask_{i}.png' for i in range(3)]:
                try:
                    os.remove(f)
                except:
                    pass

def main():
    """Main function to run the mask selector"""
    data_dir = Path("data")
    results_dir = Path("results")
    
    selector = MaskSelector(data_dir, results_dir)
    selector.process_all_images()

if __name__ == "__main__":
    main()