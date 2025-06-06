import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob


def overlay_mask(image, mask):
    """Overlay a semi-transparent mask on the image."""
    # Convert both to RGBA
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if mask.mode != "RGBA":
        mask = mask.convert("RGBA")

    # Create red overlay for mask
    mask_array = np.array(mask)
    red_mask = np.zeros_like(mask_array)
    red_mask[..., 0] = 255  # Red channel
    # Invert the mask: apply red overlay where mask is 0 (unmasked region)
    red_mask[..., 3] = (
        255 - mask_array[..., 0]
    ) * 0.5  # Alpha channel (50% transparency)

    # Composite the images
    image_array = np.array(image)
    mask_image = Image.alpha_composite(image, Image.fromarray(red_mask))
    return mask_image


def plot_ddim_results(base_dir="results/inpainting_tests"):
    # Get all image folders and filter out image_05
    image_folders = sorted(
        [
            f
            for f in glob.glob(os.path.join(base_dir, "image_*"))
            if not f.endswith("image_05")
        ]
    )
    n_rows = len(image_folders)
    n_cols = 6  # original + original with mask + 4 different T values

    # Create figure
    fig = plt.figure(
        figsize=(24, 4 * n_rows)
    )  # Increased width to accommodate new column

    # Set larger font size for titles
    plt.rcParams.update({"font.size": 14})

    for row, folder in enumerate(image_folders):
        # Load original image and mask
        original = Image.open(os.path.join(folder, "original.png"))
        mask = Image.open(os.path.join(folder, "mask.png"))

        # Plot original image without mask
        ax1 = plt.subplot(n_rows, n_cols, row * n_cols + 1)
        plt.imshow(original)
        if row == 0:
            plt.title("Original", fontsize=20, pad=15)
        plt.axis("off")
        # Add light blue background
        ax1.set_facecolor((0.9, 0.95, 1.0))  # Very light blue

        # Plot original with mask overlay
        ax2 = plt.subplot(n_rows, n_cols, row * n_cols + 2)
        plt.imshow(overlay_mask(original, mask))
        if row == 0:
            plt.title("Masked Region", fontsize=20, pad=15)
        plt.axis("off")
        # Add light blue background
        ax2.set_facecolor((0.9, 0.95, 1.0))  # Very light blue

        # Plot results for different T values
        for i, t_val in enumerate([20, 50, 100, 200]):
            result_path = os.path.join(folder, f"ddim_T{t_val}", "inpainted.png")
            if os.path.exists(result_path):
                result = Image.open(result_path)
                plt.subplot(n_rows, n_cols, row * n_cols + i + 3)
                plt.imshow(result)
                if row == 0:
                    plt.title(f"T={t_val}", fontsize=20, pad=15)
                plt.axis("off")
            else:
                print(f"Warning: Missing result for {result_path}")

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(base_dir, "ddim_comparison.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    plot_ddim_results()
