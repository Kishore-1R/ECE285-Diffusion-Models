# RePaint Implementation with DDPM and DDIM

This project implements image inpainting using diffusion models, specifically implementing the RePaint approach along with DDIM sampling. We build upon the guided-diffusion framework to create a robust inpainting solution that can handle arbitrary masks.

## Project Overview

Our implementation focuses on two key aspects:
1. **RePaint Implementation**: We implement from scratch the RePaint algorithm which uses a resampling strategy to improve inpainting quality. The key innovation is in the sampling process where we resample certain timesteps (controlled by parameter U) to better handle the inpainting mask.
2. **DDIM Integration**: In addition to RePaint's DDPM-based sampling, we also implement DDIM (Denoising Diffusion Implicit Models) sampling for faster inference while maintaining quality.

### Technical Details

The core of our implementation lies in the `gaussian_diffusion.py` module where we implement:

1. **Custom Sampling Loop**: 
   - Implements both DDPM and DDIM sampling strategies
   - Handles masked inputs for inpainting
   - Supports resampling steps (U parameter) for better mask handling
   - Uses fp16 (half precision) for faster computation

2. **Mask Handling**:
   - Integrates with Segment Anything Model (SAM) for interactive mask generation
   - Implements mask dilation to better handle object boundaries
   - Supports arbitrary mask shapes and sizes

3. **Sampling Strategies**:
   - DDPM with resampling (RePaint's core algorithm)
   - DDIM for faster sampling
   - Configurable number of sampling steps and resampling factor U

## Project Structure

```
.
├── data/                    # Directory for input images
├── results/                 # Output directory for inpainting results
│   └── inpainting_tests/   # Individual test results
├── models/                  # Pre-trained models
├── external/               # External dependencies (SAM)
├── guided-diffusion/       # Core diffusion implementation
└── src/                    # Source code
    ├── magic_eraser.py    # Main script for end users
    ├── prepare_masks.py   # Mask generation utilities
    └── run_inpainting.py  # Inpainting pipeline
```

## Quick Start

1. **Setup**:
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd [repository-name]

   # Install dependencies
   pip install -r requirements.txt

   # Download models
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

3. **Results**:
   - Results will be saved in `results/inpainting_tests/[image_name]/`
   - You'll find:
     - `mask.png`: The generated mask
     - `inpainted.png`: The final inpainted result
     - `evolution.gif`: The inpainting process visualization

## Implementation Details

### RePaint Algorithm

Our implementation of RePaint focuses on the resampling strategy for masked regions. The key components are:

1. **Resampling Strategy**:
   - Parameter U controls how many times each timestep is resampled
   - Higher U values generally lead to better results but longer computation time
   - Default U=10 provides a good balance of quality and speed

2. **Mask Handling**:
   - Known regions are preserved through the sampling process
   - Unknown (masked) regions are resampled U times
   - Gradual transition between known and unknown regions through mask dilation

### DDIM Integration

We also implement DDIM sampling which offers:
- Faster sampling compared to DDPM
- Deterministic sampling path
- Fewer required sampling steps

## Advanced Usage

For advanced users who want to experiment with different parameters:

1. **Mask Generation**:
   ```bash
   python src/prepare_masks.py
   ```
   - Interactive mask selection using SAM
   - Adjustable mask dilation

2. **Batch Processing**:
   ```bash
   python src/run_inpainting.py
   ```
   - Process multiple images
   - Test different U values
   - Compare DDPM and DDIM results

## Results

[To be added: Example results and comparisons]

## Acknowledgments

This project builds upon several key works:
- [Guided Diffusion](https://github.com/openai/guided-diffusion)
- [RePaint Paper](https://arxiv.org/abs/2201.09865)
- [Segment Anything Model](https://segment-anything.com/)