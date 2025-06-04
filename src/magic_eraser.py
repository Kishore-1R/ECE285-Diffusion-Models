import sys
import os
from pathlib import Path
import time

import global_vars as gv

# Add the project root and guided-diffusion to Python path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))
guided_diffusion_path = current_dir / "guided-diffusion"
sys.path.append(str(guided_diffusion_path))

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
    model_config.update(
        {
            "attention_resolutions": "32,16,8",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "1000",
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )

    # Create and load model
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        dist_util.load_state_dict(
            "../codes/models/256x256_diffusion_uncond.pt", map_location="cpu"
        )
    )
    model.convert_to_fp16()
    model.to(dist_util.dev())
    model.eval()

    return model, diffusion


def process_image(
    image_path,
    selector,
    model,
    diffusion,
    results_dir,
    sampler_type="ddpm",
    T_sampling=1000,
    U=10,
):
    """Process a single image with user verification"""
    print(f"\nProcessing {image_path.name}")

    print("\nInstructions for object removal:")
    print("1. A new png image will be created")
    print("2. Select points on the object you want to remove")
    print("   - Select points inside the object (default: positive points)")
    print("   - Add negative points outside by typing 'x,y,0' if needed")
    print("3. Type 'done' when you're satisfied with the points")
    print("4. Three mask options (0-2) will be saved as 'mask_[0-2].png'")
    print("5. Review these masks and select the best one (0-2)")
    print("6. Type 'retry' if you want to try again")
    print("7. Type 'quit' to exit the program")

    while True:
        # Generate mask
        if selector.process_single_image(image_path) == "quit":
            print("\nExiting program...")
            sys.exit(0)

        # Give user time to check the masks
        print("\nMask previews have been saved. Please check:")
        print("- mask_0.png: First mask option")
        print("- mask_1.png: Second mask option")
        print("- mask_2.png: Third mask option")

        proceed = input(
            "\nAre you satisfied with the selected mask? (yes/no): "
        ).lower()
        if proceed == "yes":
            # Run inpainting
            test_dir = results_dir / "inpainting_tests" / image_path.stem
            if test_dir.exists():
                print(f"\nRunning inpainting with {sampler_type.upper()}...")
                print("The process cannot be interrupted without losing progress.")
                run_inpainting(
                    test_dir,
                    model,
                    diffusion,
                    device=dist_util.dev(),
                    sampler_type=sampler_type,
                    U=U,
                    eta=gv.eta,
                    T_sampling=T_sampling,
                )
                print(f"\nResults saved in: {test_dir}")
                print("- mask.png: The mask used for removal")
                print("- inpainted.png: The final result")
                print("- evolution.gif: The inpainting process")
                break
        else:
            print("\nLet's try selecting the object again...")
            # Clean up temporary files
            for f in ["current_image.png"] + [f"mask_{i}.png" for i in range(3)]:
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

    print(f"\nAvailable images in data/ directory:")
    for img in image_files:
        print(f"- {img.name}")

    # Ask user for specific image
    while True:
        filename = input(
            "\nEnter the filename (with extension) of the image to process: "
        )
        image_path = data_dir / filename
        if image_path in image_files:
            break
        print(f"Error: {filename} not found in data/ directory. Please try again.")

    # Ask user for sampler choice
    while True:
        sampler_choice = input("\nChoose sampler type (ddpm/ddim): ").lower()
        if sampler_choice in ["ddpm", "ddim"]:
            break
        print("Error: Please enter either 'ddpm' or 'ddim'")

    # Set sampling parameters based on choice
    if sampler_choice == "ddpm":
        print("\nNote: DDPM requires 1000 sampling steps and may take a while.")
        proceed = input("Do you want to continue? (yes/no): ").lower()
        if proceed != "yes":
            print("\nExiting program...")
            return
        T_sampling = 1000
        U = 10  # Default U value for DDPM
    else:  # DDIM
        while True:
            try:
                T_sampling = int(
                    input(
                        "\nEnter number of sampling steps for DDIM (recommended: less than 50 for speedup): "
                    )
                )
                if T_sampling > 0:
                    break
                print("Error: Number of steps must be positive.")
            except ValueError:
                print("Error: Please enter a valid number.")

    # Ask for U value (resampling steps)
    while True:
        try:
            U = int(input("\nEnter number of resampling steps (U) (recommended: 10): "))
            if U > 0:
                break
            print("Error: U must be positive.")
        except ValueError:
            print("Error: Please enter a valid number.")

    # Check if mask and original image already exist
    test_dir = results_dir / "inpainting_tests" / image_path.stem
    mask_path = test_dir / "mask.png"
    original_path = test_dir / "original.png"

    use_existing = False
    if mask_path.exists() and original_path.exists():
        print("\nFound existing mask and original image.")
        proceed = input("Would you like to use the existing mask? (yes/no): ").lower()
        if proceed == "yes":
            use_existing = True

    # Load diffusion model
    print("\nLoading diffusion model...")
    model, diffusion = setup_model()

    # Only initialize SAM if we're not using existing mask
    selector = None
    if not use_existing:
        print("\nInitializing Segment Anything Model...")
        selector = MaskSelector(data_dir, results_dir, dilation_size=5)

    # Process the selected image
    if use_existing:
        print(f"\nRunning inpainting with {sampler_choice.upper()}...")
        print("The process cannot be interrupted without losing progress.")
        run_inpainting(
            test_dir,
            model,
            diffusion,
            device=dist_util.dev(),
            sampler_type=sampler_choice,
            U=U,
            eta=gv.eta,
            T_sampling=T_sampling,
        )
        print(f"\nResults saved in: {test_dir}")
        print("- mask.png: The mask used for removal")
        print("- inpainted.png: The final result")
        print("- evolution.gif: The inpainting process")
    else:
        process_image(
            image_path,
            selector,
            model,
            diffusion,
            results_dir,
            sampler_type=sampler_choice,
            T_sampling=T_sampling,
            U=U,
        )

    print("\nImage processed successfully!")
    print("Check the results/ directory for your processed image.")


if __name__ == "__main__":
    main()
