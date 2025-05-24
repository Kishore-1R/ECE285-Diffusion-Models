import torch

def generate_random_mask(shape, pixel_ratio):
    """
    Generates a random binary mask with the given pixel ratio.

    Args:
        shape (tuple): Shape of the mask (B, C, H, W).
        pixel_ratio (float): Ratio of pixels to be set to 1.

    Returns:
        torch.Tensor: Random binary mask.
    """
    C, H, W = shape

    num_pixels = H * W
    num_ones = int(num_pixels * pixel_ratio)
    
    # Generate a flat array with the appropriate ratio of ones and zeros
    flat_mask = torch.zeros(num_pixels, dtype=torch.float32)
    flat_mask[:num_ones] = 1
    
    # Shuffle to randomize the positions of ones and zeros
    flat_mask = flat_mask[torch.randperm(num_pixels)]
    
    # Reshape to the original spatial dimensions and duplicate across channels
    mask = flat_mask.view(1, H, W)
    mask = mask.expand(C, H, W)
    
    return mask


def generate_center_mask(shape, box_size=100):
    """
    Generates a binary mask with a centered square of ones.

    Args:
        shape (tuple): Shape of the mask (C, H, W) or (1, C, H, W).
        box_size (int): Size of the central square to unmask (1s). Default: 100

    Returns:
        torch.Tensor: Binary mask of shape (C, H, W)
    """
    if len(shape) == 4:
        _, C, H, W = shape
    elif len(shape) == 3:
        C, H, W = shape
    else:
        raise ValueError("Unsupported shape: expected (C, H, W) or (1, C, H, W)")

    mask = torch.ones((C, H, W), dtype=torch.float32)

    y0 = (H - box_size) // 2
    x0 = (W - box_size) // 2
    y1 = y0 + box_size
    x1 = x0 + box_size

    mask[:, y0:y1, x0:x1] = 0.0
    return mask
