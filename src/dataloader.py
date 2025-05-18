import cv2
import numpy as np
import torch
import os


def load_and_resize_image(img_path, save_resized=False, save_path=None):
    """
    Loads an image, resizes to 256x256, and returns a torch tensor (C, H, W) normalized to [0,1].

    Args:
        img_path (str): Path to input image (.jpg or .png)
        save_resized (bool): If True, saves the resized image.
        save_path (str): Path to save resized image (optional)

    Returns:
        torch.Tensor: Float tensor of shape (3, 256, 256), range [0, 1]
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    if save_resized:
        if save_path is None:
            save_path = os.path.splitext(img_path)[0] + "_resized.png"
        cv2.imwrite(save_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return img_tensor


def save_tensor_as_image(tensor, save_path):
    """
    Saves a PyTorch tensor as an image file using OpenCV.

    Args:
        tensor (torch.Tensor or np.ndarray): Image tensor. Shape must be [3, H, W] or [H, W, 3].
        save_path (str): Output file path (e.g., 'output.png')
    """
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)  # [H, W, C]
        img_np = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        img_np = tensor
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")

    # Assume float image in [0, 1] — rescale to [0, 255] if needed
    if img_np.max() <= 1.0:
        img_np = img_np * 255.0

    img_np = img_np.clip(0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
