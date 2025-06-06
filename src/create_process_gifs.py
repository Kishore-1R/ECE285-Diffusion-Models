import os
import numpy as np
from PIL import Image
import imageio
from pathlib import Path


def create_process_gif(image_dir, result_subdir, output_name="process.gif", fps=2):
    """Create a GIF showing the full inpainting process.

    Steps:
    1. Show original image
    2. Show mask overlay
    3. Show masked image (with region removed)
    4. Show evolution of inpainting (only in the masked region)

    The GIF will loop endlessly.
    """
    image_dir = Path(image_dir)

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

    # Create masked image (region removed)
    masked_np = original_np * mask_np[..., None]

    # Prepare initial frames
    frames = []
    # Show original for 2 seconds (2 * fps frames)
    frames.extend([original_np] * (2 * fps))
    # Show overlay for 2 seconds
    frames.extend([overlay_np.astype(np.uint8)] * (2 * fps))
    # Show masked image for 2 seconds
    frames.extend([masked_np.astype(np.uint8)] * (2 * fps))

    # Load and append evolution frames
    evolution_path = image_dir / result_subdir / "evolution.gif"
    if evolution_path.exists():
        evolution_frames = imageio.mimread(evolution_path)
        # Convert evolution frames to uint8 if needed
        evolution_frames = [
            (
                (frame * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                if frame.dtype != np.uint8
                else frame
            )
            for frame in evolution_frames
        ]

        # Create composite frames showing evolution only in the masked region
        for evolution_frame in evolution_frames:
            # Create a composite where:
            # - masked_np shows in the unmasked region (where mask is 1)
            # - evolution_frame shows in the masked region (where mask is 0)
            composite = masked_np.copy()
            mask_3d = mask_np[..., None]  # Add channel dimension
            composite = composite * mask_3d + evolution_frame * (1 - mask_3d)
            frames.append(composite.astype(np.uint8))
    else:
        print(f"Warning: Evolution GIF not found at {evolution_path}")

    # Save the combined GIF with infinite looping
    output_path = image_dir / output_name
    imageio.mimsave(output_path, frames, fps=fps, loop=0)  # loop=0 means infinite loop
    print(f"Saved endlessly looping process GIF to {output_path}")


def main():
    # Define the cases to process
    cases = [
        ("image_04", "ddim_T20"),
        ("image_09", "ddim_U10_T20"),
        ("image_08", "ddim_T20"),
        ("image_10", "ddim_U10_T20")
    ]

    base_dir = Path("results/inpainting_tests")

    for image_folder, result_subdir in cases:
        print(f"\nProcessing {image_folder} with {result_subdir}...")
        image_dir = base_dir / image_folder
        if not image_dir.exists():
            print(f"Warning: Directory not found: {image_dir}")
            continue

        create_process_gif(image_dir, result_subdir)


if __name__ == "__main__":
    main()
