import os
import numpy as np
from PIL import Image
import imageio
from pathlib import Path


def create_summary_gif(cases, output_name="results_summary.gif", fps=4):
    """Create a GIF showing all results in sequence.

    For each case, shows:
    1. Original image
    2. Mask overlay
    3. Final inpainted result

    The GIF will loop endlessly.
    """
    base_dir = Path("results/inpainting_tests")
    frames = []

    for image_folder, result_subdir in cases:
        print(f"\nProcessing {image_folder} with {result_subdir}...")
        image_dir = base_dir / image_folder

        if not image_dir.exists():
            print(f"Warning: Directory not found: {image_dir}")
            continue

        # Load original image and mask
        original = Image.open(image_dir / "original.png")
        mask = Image.open(image_dir / "mask.png")

        # Convert to numpy arrays and normalize
        original_np = np.array(original)
        mask_np = np.array(mask) / 255.0  # Convert to [0, 1]

        # Create overlay frame
        overlay_np = original_np.copy()
        # Create red overlay for the region to be inpainted (where mask is 0)
        red_overlay = np.zeros_like(original_np)
        red_overlay[..., 0] = 255  # Red channel
        # Blend where mask is 0 (inpainting region)
        alpha = 0.5 * (1 - mask_np[..., None])  # 50% transparency
        overlay_np = (1 - alpha) * original_np + alpha * red_overlay

        # Load inpainted result
        inpainted_path = image_dir / result_subdir / "inpainted.png"
        if not inpainted_path.exists():
            print(f"Warning: Inpainted result not found at {inpainted_path}")
            continue
        inpainted = np.array(Image.open(inpainted_path))

        # Add frames for this case
        # Show each frame for 0.75 seconds (0.75 * fps frames)
        frames.extend([original_np] * (int(0.75 * fps)))  # Original
        frames.extend([overlay_np.astype(np.uint8)] * (int(0.75 * fps)))  # Overlay
        frames.extend([inpainted] * (int(0.75 * fps)))  # Result

    # Save the combined GIF with infinite looping
    output_path = base_dir / output_name
    imageio.mimsave(output_path, frames, fps=fps, loop=0)  # loop=0 means infinite loop
    print(f"\nSaved endlessly looping summary GIF to {output_path}")


def main():
    # Define the cases to process
    cases = [
        ("image_04", "ddim_T20"),
        ("image_09", "ddim_U10_T20"),
        ("image_08", "ddim_T20"),
        ("image_01", "ddpm_U10"),
        ("image_06", "ddim_T20"),
        ("finger", "ddim_U10_T20"),
        ("image_02", "ddim_T20"),
    ]

    create_summary_gif(cases)


if __name__ == "__main__":
    main()
