import sys
import os
from pathlib import Path

# Add the parent directory to Python path to find guided_diffusion
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import numpy as np
import torch
from PIL import Image
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
import imageio
from tqdm import tqdm
import argparse


def load_and_resize(image_path, mask_path, image_size=256):
    """Load and resize image and mask to square"""
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Resize maintaining aspect ratio and center crop
    def resize_with_pad(im, desired_size):
        old_size = im.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.Resampling.LANCZOS)

        # Center pad to desired size
        new_im = Image.new(
            "RGB" if im.mode == "RGB" else "L",
            (desired_size, desired_size),
            "white" if im.mode == "L" else (0, 0, 0),
        )
        new_im.paste(
            im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
        )
        return new_im

    image = resize_with_pad(image, image_size)
    mask = resize_with_pad(mask, image_size)

    # Convert to numpy arrays and normalize
    image_np = np.array(image).astype(np.float32) / 127.5 - 1
    mask_np = np.array(mask).astype(np.float32) / 255.0

    return image_np, mask_np


def save_gif(frames, path, fps=10):
    """Save frames as gif"""
    # Sample evenly spaced frames (total 50 frames)
    if len(frames) > 50:
        indices = np.linspace(0, len(frames) - 1, 50, dtype=int)
        frames = [frames[i] for i in indices]

    # Convert to uint8
    frames = [(frame * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for frame in frames]
    imageio.mimsave(path, frames, fps=fps)


def run_inpainting(
    output_dir,
    model,
    diffusion,
    device,
    U=1,
    T_sampling=1000,
    image_path=None,
    mask_path=None,
):
    """Run inpainting with DDPM sampling

    Args:
        output_dir: Directory where results will be saved (will be created as image_xx/ddpm_U{U}/)
        image_path: Path to the original image
        mask_path: Path to the mask image
        U: Number of update steps per noise level
        T_sampling: Number of sampling steps
    """
    # Load image and mask from the correct paths
    if image_path is None or mask_path is None:
        raise ValueError("Both image_path and mask_path must be provided")

    # Verify files exist
    if not image_path.exists():
        raise FileNotFoundError(f"Original image not found at {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found at {mask_path}")

    image, mask = load_and_resize(image_path, mask_path)

    # Convert to torch tensors with half precision
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).half()
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).half()

    # Move to device
    image = image.to(device)
    mask = mask.to(device)

    # Setup sampling parameters
    model_kwargs = {
        "sampler_type": "ddpm",
        "U": U,
        "T_sampling": T_sampling,
    }

    # Run diffusion (keep in fp16)
    sample, xs, _ = diffusion.custom_sample_loop(
        model,
        shape=image.shape,
        noise=None,
        measurement=image,
        mask=mask,
        model_kwargs=model_kwargs,
        device=device,
        progress=True,
    )

    # Create output directory if it doesn't exist
    output_dir = output_dir / f"ddpm_U{U}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save final sample
    sample_np = sample.cpu().numpy().squeeze().transpose(1, 2, 0)
    sample_image = Image.fromarray(
        ((sample_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    )
    sample_image.save(output_dir / "inpainted.png")

    # Save evolution gif
    frames = [x.cpu().numpy().squeeze().transpose(1, 2, 0) for x in xs]
    save_gif(frames, output_dir / "evolution.gif")

    return sample_np


def main():
    # Load model
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

    # Load model
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        dist_util.load_state_dict(
            "../codes/models/256x256_diffusion_uncond.pt", map_location="cpu"
        )
    )
    model.convert_to_fp16()
    model.to(dist_util.dev())
    model.eval()

    # Get test directories for images 01 to 09
    results_dir = Path("results") / "inpainting_tests"
    test_dirs = []
    for i in range(1, 10):  # 01 to 09
        img_dir = results_dir / f"image_{i:02d}"
        if img_dir.exists():
            test_dirs.append(img_dir)
        else:
            print(f"Warning: Directory {img_dir} not found")
    test_dirs.sort()

    # Run DDPM sampling with different U values
    U_values = [1, 2, 5, 10]
    for test_dir in test_dirs:
        print(f"\nProcessing {test_dir.name}")

        # Find image and mask files
        image_file = test_dir / "original.png"
        mask_file = test_dir / "mask.png"

        if not image_file.exists() or not mask_file.exists():
            print(f"Skipping {test_dir.name} - missing image or mask files")
            continue

        for U in U_values:
            print(f"Running DDPM sampling with U={U}")
            run_inpainting(
                test_dir,
                model,
                diffusion,
                device=dist_util.dev(),
                U=U,
                image_path=image_file,
                mask_path=mask_file,
            )


if __name__ == "__main__":
    main()
