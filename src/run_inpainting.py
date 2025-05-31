import sys
import os
from pathlib import Path
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
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        
        # Center pad to desired size
        new_im = Image.new("RGB" if im.mode == "RGB" else "L", (desired_size, desired_size), "white" if im.mode == "L" else (0,0,0))
        new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
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
        indices = np.linspace(0, len(frames)-1, 50, dtype=int)
        frames = [frames[i] for i in indices]
    
    # Convert to uint8
    frames = [(frame * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for frame in frames]
    imageio.mimsave(path, frames, fps=fps)

def run_inpainting(image_dir, model, diffusion, device, sampler_type="ddpm", U=1):
    """Run inpainting for a single configuration"""
    # Load image and mask
    image_path = image_dir / "original.png"
    mask_path = image_dir / "mask.png"
    image, mask = load_and_resize(image_path, mask_path)
    
    # Convert to torch tensors with half precision
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).half()  # Convert to float16
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).half()  # Convert to float16
    
    # Move to device
    image = image.to(device)
    mask = mask.to(device)
    
    # Setup sampling parameters
    model_kwargs = {
        "sampler_type": sampler_type,
        "U": U,
        "eta": 0.15,
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
        progress=True
    )
    
    # Save results
    result_dir = image_dir / f"{sampler_type}_U{U}"
    result_dir.mkdir(exist_ok=True)
    
    # Save final sample (stay in fp16 until numpy conversion)
    sample_np = sample.cpu().numpy().squeeze().transpose(1, 2, 0)
    sample_image = Image.fromarray(((sample_np + 1) * 127.5).clip(0, 255).astype(np.uint8))
    sample_image.save(result_dir / "inpainted.png")
    
    # Save evolution gif (stay in fp16 until numpy conversion)
    frames = [x.cpu().numpy().squeeze().transpose(1, 2, 0) for x in xs]
    save_gif(frames, result_dir / "evolution.gif")
    
    return sample_np

def parse_args():
    parser = argparse.ArgumentParser(description='Run inpainting with resume capability')
    parser.add_argument('--resume_image', type=str, help='Resume from this image number (e.g., "07")')
    parser.add_argument('--resume_u', type=int, help='Resume from this U value')
    parser.add_argument('--resume_sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'], 
                       help='Resume from this sampler type')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32,16,8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '1000',  # Can be adjusted for faster sampling
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
    
    # Load model
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        dist_util.load_state_dict("../codes/models/256x256_diffusion_uncond.pt", map_location="cpu")
    )
    model.convert_to_fp16()  # Convert model to fp16 before moving to device
    model.to(dist_util.dev())
    model.eval()
    
    # Get all test directories in order
    results_dir = Path("results") / "inpainting_tests"
    test_dirs = sorted(list(results_dir.glob("image_*")))
    
    # Define all configurations to run
    configs = []
    for u in [1, 2, 5, 10]:
        for test_dir in test_dirs:
            configs.append({"dir": test_dir, "U": u, "sampler": "ddpm"})
    # Add DDIM configs
    for test_dir in test_dirs:
        configs.append({"dir": test_dir, "U": 10, "sampler": "ddim"})
    
    # Find where to resume from
    if args.resume_image and args.resume_u:
        resume_idx = 0
        for idx, config in enumerate(configs):
            img_num = int(config["dir"].name.split('_')[1])
            if (img_num == int(args.resume_image) and 
                config["U"] == args.resume_u and 
                config["sampler"] == args.resume_sampler):
                resume_idx = idx
                break
        configs = configs[resume_idx:]
    
    # Run all configurations
    for config in configs:
        print(f"\nProcessing {config['dir'].name} with {config['sampler'].upper()} U={config['U']}")
        run_inpainting(
            config['dir'],
            model,
            diffusion,
            device=dist_util.dev(),
            sampler_type=config['sampler'],
            U=config['U']
        )

if __name__ == "__main__":
    main()