import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms
import cv2
import io


def print_image_stats(img, stage=""):
    """Print detailed statistics about the image"""
    print(f"\n{stage} image stats:")
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Min: {img.min()}")
    print(f"Max: {img.max()}")
    print(f"Mean: {img.mean():.2f}")
    if len(img.shape) == 3:
        print("Channel stats:")
        for i, channel in enumerate(["R", "G", "B"]):
            print(
                f"{channel} - Min: {img[...,i].min()}, Max: {img[...,i].max()}, Mean: {img[...,i].mean():.2f}"
            )


def apply_tone_mapping(img):
    """Apply tone mapping to convert HDR to SDR"""
    # Convert to float32 for processing
    img_float = img.astype(np.float32) / 255.0

    # Create a Reinhard tone mapping operator
    tonemap = cv2.createTonemapReinhard(
        gamma=1.0, intensity=0.0, light_adapt=0.8, color_adapt=0.0
    )

    # Apply tone mapping
    img_tone_mapped = tonemap.process(img_float)

    # Convert back to uint8
    return (img_tone_mapped * 255).clip(0, 255).astype(np.uint8)


def load_image_direct(image_path):
    """Load image with HDR handling"""
    try:
        with Image.open(image_path) as img:
            # Get all image info
            icc_profile = img.info.get("icc_profile")
            exif = img.info.get("exif")

            # Print image info
            print(f"\nOriginal image mode: {img.mode}")
            print(f"ICC Profile present: {icc_profile is not None}")
            print(f"EXIF data present: {exif is not None}")
            print(f"Original size: {img.size}")

            if icc_profile:
                try:
                    # Try to read the ICC profile
                    icc = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
                    print(f"ICC Profile Description: {icc.profile.profile_description}")
                    print(f"ICC Profile Copyright: {icc.profile.copyright}")
                    print(f"ICC Profile Color Space: {icc.profile.color_space}")
                except Exception as e:
                    print(f"Failed to read ICC profile details: {e}")

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to numpy array
            img_array = np.array(img)

            return img_array, icc_profile, exif

    except Exception as e:
        print(f"PIL failed to load image: {e}")
        return None, None, None


def save_image(image, save_path, icc_profile=None, exif=None):
    """Save image with different HDR handling methods"""
    if image.dtype != np.uint8:
        image = (image.clip(0, 255)).astype(np.uint8)

    base_path = Path(save_path)

    # 1. Save original with metadata
    pil_img = Image.fromarray(image)
    if icc_profile or exif:
        pil_img.save(save_path, icc_profile=icc_profile, exif=exif)
    else:
        pil_img.save(save_path)

    # 2. Save with tone mapping
    tone_mapped = apply_tone_mapping(image)
    Image.fromarray(tone_mapped).save(
        base_path.parent / f"{base_path.stem}_tone_mapped{base_path.suffix}"
    )

    # 3. Save with OpenCV's HDR conversion
    cv2_path = base_path.parent / f"{base_path.stem}_cv2{base_path.suffix}"
    # Convert to float32 and normalize
    img_float32 = image.astype(np.float32) / 255.0
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img_float32, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(np.uint8(l * 255)) / 255.0
    lab = cv2.merge([l, a, b])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    cv2.imwrite(
        str(cv2_path), cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )


def main():
    # Get image path from command line
    if len(sys.argv) != 2:
        print("Usage: python debug_image.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image {image_path} not found")
        sys.exit(1)

    print(f"\nProcessing {image_path}")

    # Load image directly without resizing
    image, icc_profile, exif = load_image_direct(image_path)
    if image is None:
        print("Failed to load image")
        sys.exit(1)

    print_image_stats(image, "After direct load")

    # Save with _test suffix
    save_path = image_path.parent / f"{image_path.stem}_test{image_path.suffix}"

    # Save with different HDR handling methods
    save_image(image, save_path, icc_profile, exif)

    # Read back and check the saved images
    print("\nChecking different saved versions:")
    for suffix in ["test", "tone_mapped", "cv2"]:
        check_path = (
            image_path.parent / f"{image_path.stem}_{suffix}{image_path.suffix}"
        )
        with Image.open(check_path) as saved_img:
            saved_array = np.array(saved_img)
            print_image_stats(saved_array, f"After saving and reloading ({suffix})")

    print(f"\nSaved test images to:")
    print(
        f"1. {image_path.parent / f'{image_path.stem}_test{image_path.suffix}' } (with original metadata)"
    )
    print(
        f"2. {image_path.parent / f'{image_path.stem}_tone_mapped{image_path.suffix}' } (with HDR tone mapping)"
    )
    print(
        f"3. {image_path.parent / f'{image_path.stem}_cv2{image_path.suffix}' } (with CLAHE enhancement)"
    )


if __name__ == "__main__":
    main()
