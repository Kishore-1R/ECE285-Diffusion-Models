# RePaint Implementation with DDIM Sampling

This branch implements image inpainting using diffusion models with a focus on DDIM (Denoising Diffusion Implicit Models) sampling for faster inference. We build upon the guided-diffusion framework and integrate SAM (Segment Anything Model) for interactive mask generation.

## Branch Overview

This DDIM branch offers:
1. **Fast Inference**: Using DDIM sampling with configurable steps (20-200 steps vs 1000 steps in DDPM)
2. **Quality Control**: Adjustable parameters for balancing speed and quality
3. **Interactive Masking**: Integration with SAM for easy object removal

### Key Features

1. **DDIM Sampling**:
   - Configurable number of sampling steps (T=20 to 200)
   - Adjustable number of iterations (U=1 to 10)
   - Deterministic sampling path
   - Significantly faster than DDPM (which requires 1000 steps)

2. **Interactive Interface**:
   - Simple command-line interface
   - Visual feedback for mask selection
   - Multiple mask options for best results
   - Progress visualization through evolution GIFs

## Quick Start

1. **Setup**:
   ```bash
   # Clone the repository and switch to DDIM branch
   git clone [repository-url]
   cd [repository-name]
   git checkout ddim

   # Install dependencies
   pip install -r requirements.txt

   # Download required models
   # - SAM checkpoint (sam_vit_h_4b8939.pth) should be in models/
   # - Diffusion model (256x256_diffusion_uncond.pt) should be in models/
   ```

2. **Usage**:
   ```bash
   # Place your image in the data/ directory
   cp your_image.jpg data/

   # Run the magic eraser
   python src/magic_eraser.py
   ```

   The script will prompt you for:
   - Image selection
   - Number of sampling steps (T) - recommended range: 20-200
   - Number of iterations (U) - recommended range: 1-10

3. **Results**:
   Results will be saved in `results/inpainting_tests/[image_name]/`:
   - `mask.png`: The generated mask
   - `inpainted.png`: The final inpainted result
   - `evolution.gif`: Visualization of the inpainting process

## Parameter Guide

### DDIM Parameters

1. **Sampling Steps (T)**:
   - Range: 20-200 steps
   - Lower values (20-50): Faster but potentially lower quality
   - Higher values (100-200): Better quality but slower
   - Default recommendation: 50 steps for a good speed/quality balance

2. **Iterations (U)**:
   - Range: 1-10 iterations
   - Lower values (1-3): Faster, suitable for simple masks
   - Higher values (5-10): Better for complex masks
   - Default recommendation: U=5 for most cases

### Mask Generation

The SAM-based mask generation offers:
- Interactive point selection
- Multiple mask options
- Adjustable mask dilation
- Option to retry if not satisfied

## Comparison with DDPM Branch

This DDIM branch differs from the main (DDPM) branch in several ways:

| Feature           | DDIM Branch (This)     | DDPM Branch (Main)     |
|------------------|------------------------|----------------------|
| Sampling Steps   | 20-200 (configurable) | Fixed 1000 steps    |
| Speed            | 5-50x faster          | Standard speed      |
| Deterministic    | Yes                   | No                  |
| Memory Usage     | Lower                 | Higher              |
| Quality         | Good                  | Slightly better     |

## Advanced Usage

For advanced users who want to experiment with different parameters:

1. **Batch Testing**:
   ```bash
   python src/run_ddim_test.py
   ```
   - Test multiple T values (20, 50, 100, 200)
   - Compare results across different parameters
   - Automated processing of multiple images

## Results

[To be added: Example results showing DDIM performance at different T values]

## Acknowledgments

This project builds upon several key works:
- [Guided Diffusion](https://github.com/openai/guided-diffusion)
- [DDIM Paper](https://arxiv.org/abs/2010.02502)
- [RePaint Paper](https://arxiv.org/abs/2201.09865)
- [Segment Anything Model](https://segment-anything.com/)