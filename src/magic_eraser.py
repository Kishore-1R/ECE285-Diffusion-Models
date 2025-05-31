import sys
import os
from pathlib import Path
import time

# Add the project root to Python path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

# Import from our modules
from prepare_masks import MaskSelector
from run_inpainting import load_and_resize, run_inpainting
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

def setup_model():
    """Setup and load the diffusion model"""
    # Load model configuration
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32,16,8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '1000',
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
    
    # Create and load model
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        dist_util.load_state_dict("models/256x256_diffusion_uncond.pt", map_location="cpu")
    )
    model.convert_to_fp16()
    model.to(dist_util.dev())
    model.eval()
    
    return model, diffusion

def process_image(image_path, selector, model, diffusion):
    """Process a single image with user verification"""
    print(f"\nProcessing {image_path.name}")
    print("\nInstructions for object removal:")
    print("1. A window with your image will appear")
    print("2. Click points on the object you want to remove")
    print("   - Click points inside the object (default: positive points)")
    print("   - Add negative points outside by typing 'x,y,0' if needed")
    print("3. Type 'done' when you're satisfied with the points")
    print("4. Three mask options (0-2) will be saved as 'mask_[0-2].png'")
    print("5. Review these masks and select the best one (0-2)")
    print("6. Type 'retry' if you want to try again")
    
    while True:
        # Generate mask
        selector.process_single_image(image_path)
        
        # Give user time to check the masks
        print("\nMask previews have been saved. Please check:")
        print("- mask_0.png: First mask option")
        print("- mask_1.png: Second mask option")
        print("- mask_2.png: Third mask option")
        
        proceed = input("\nAre you satisfied with the selected mask? (yes/no): ").lower()
        if proceed == 'yes':
            # Run inpainting
            test_dir = results_dir / "inpainting_tests" / image_path.stem
            if test_dir.exists():
                print("\nRunning inpainting with DDPM...")
                print("The process cannot be interrupted without losing progress.")
                run_inpainting(
                    test_dir,
                    model,
                    diffusion,
                    device=dist_util.dev(),
                    sampler_type="ddpm",
                    U=10
                )
                print(f"\nResults saved in: {test_dir}")
                print("- mask.png: The mask used for removal")
                print("- inpainted.png: The final result")
                print("- evolution.gif: The inpainting process")
                break
        else:
            print("\nLet's try selecting the object again...")
            # Clean up temporary files
            for f in ['current_image.png'] + [f'mask_{i}.png' for i in range(3)]:
                try:
                    os.remove(f)
                except:
                    pass

def main():
    """Main function to run the magic eraser"""
    print("Welcome to Magic Eraser!")
    print("\nThis tool will help you remove objects from your images.")
    
    # Setup paths
    data_dir = Path("data")
    results_dir = Path("results")
    
    # Check for images in data directory
    image_files = sorted(list(data_dir.glob("*.[pj][np][g]*")))
    if not image_files:
        print("\nError: No images found in the data/ directory!")
        print("Please place your images in the data/ directory and run again.")
        return
    
    print(f"\nFound {len(image_files)} images in data/ directory:")
    for img in image_files:
        print(f"- {img.name}")
    
    # Initialize mask selector
    print("\nInitializing Segment Anything Model...")
    selector = MaskSelector(data_dir, results_dir, dilation_size=5)
    
    # Load diffusion model
    print("\nLoading diffusion model...")
    model, diffusion = setup_model()
    
    # Process each image
    for image_path in image_files:
        process_image(image_path, selector, model, diffusion)
    
    print("\nAll images processed successfully!")
    print("Check the results/ directory for your processed images.")

if __name__ == "__main__":
    main() 