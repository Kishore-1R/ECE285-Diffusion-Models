"""
Sample from the diffusion model conditioning on the available image
"""

import argparse
import os
import numpy as np
from einops import rearrange
import torch as th
import torch.distributed as dist
import random
from PIL import Image

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "guided_diffusion"))
)
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from mask_generator import generate_random_mask, generate_center_mask
from dataloader import (
    load_and_resize_image,
    save_tensor_as_image,
    save_tensor_to_gif,
)


def unnormalize_img(t):
    # Unnormalize the image tensor to the range [0, 1]
    # Assuming the input tensor is in the range [-1, 1]
    return (t + 1) * 0.5


def normalize_image(t):
    # Normalize the image tensor to the range [-1. 1]
    # Assuming the input tensor is in the range [0. 1]
    return (2 * t) - 1


def set_random_seed(seed):
    # Set the random seed for Python's built-in random module
    random.seed(seed)

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for PyTorch
    th.manual_seed(seed)

    # If you are using CUDA
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # If you are using multi-GPU.

    # For deterministic operations (not necessary for all use cases)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    # logger.configure()
    device = dist_util.dev()

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Load image
    img_path = "data/elephant.jpg"
    img_tensor = load_and_resize_image(img_path, save_resized=True).to(device)
    img_tensor = normalize_image(img_tensor)
    print("Loaded image successfully...")  # C, H, W

    # # Generate random mask
    # mask = generate_random_mask(img_tensor.shape, pixel_ratio=0.5).to(device)
    # maskedInput = mask * img_tensor
    # save_tensor_as_image(maskedInput, "data/masked_sample.png")

    # Generate square mask
    mask = generate_center_mask(img_tensor.shape).to(device)
    maskedInput = mask * img_tensor
    save_tensor_as_image(maskedInput, "data/masked_center_sample.png")

    model_kwargs = {}
    if args.class_cond:
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes

    sample, x_evol = diffusion.custom_sample_loop(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        measurement=maskedInput,
        mask=mask,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
    )

    # === Save Final Sample Image ===
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    sample = unnormalize_img(th.clamp(sample[0], -1, 1))  # sample was B, T, H, W
    save_tensor_as_image(sample, os.path.join(save_dir, "sample.png"))
    
    # === Save Evolution GIF ===
    x_evol = [x_tensor.cpu() for x_tensor in x_evol]
    preds_tensor = th.cat(x_evol, dim=0)  # shape: (T, 3, H, W)
    preds_tensor = rearrange(preds_tensor, "t c h w -> c t h w")
    preds_tensor = th.clamp(preds_tensor, -1, 1)
    save_tensor_to_gif(
        unnormalize_img(preds_tensor), os.path.join(save_dir, "evolution.gif")
    )

    # No need to gather full array or save .npz unless required
    dist.barrier()
    print("Sampling and saving complete.")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
