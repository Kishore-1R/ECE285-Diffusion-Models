"""
Sample from the diffusion model conditioning on the available image
"""

import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import random

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
from mask_generator import generate_random_mask
from dataloader import load_and_resize_image, save_tensor_as_image


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
    print("Loaded image successfully...")

    # Generate random mask
    mask = generate_random_mask(img_tensor.shape, pixel_ratio=0.5).to(device)
    save_tensor_as_image(mask * img_tensor, "data/masked_sample.png")

    model_kwargs = {}
    if args.class_cond:
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
    # sample_fn = (
    #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    # )

    sample = sample_fn(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
    )
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()

    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
    all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    if args.class_cond:
        gathered_labels = [
            th.zeros_like(classes) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
    logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


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
