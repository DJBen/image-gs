# Metal TopK Normalization Implementation

## Overview
This document describes the TopK normalization feature implementation for the Metal shader renderer in Image-GS.

## What is TopK Normalization?

TopK normalization is an advanced rendering mode in Image-GS that:

1. **Selects Top-K Contributors**: For each pixel, instead of accumulating contributions from all Gaussians, it identifies the K Gaussians with the highest alpha (influence) values.

2. **Normalizes by Sum**: The contributions are normalized by the sum of the top-K alpha values, ensuring the total weight sums to approximately 1.0.

3. **Improves Quality**: This approach can produce better visual quality by focusing on the most important contributors and avoiding numerical issues from summing many small contributions.

## Implementation Details

### Constants
```metal
constant uint MAX_TOPK = 10;  // Matches CUDA config.h
constant float EPS = 1e-4;    // Epsilon for numerical stability
```

### Two Rendering Modes

#### 1. Basic Mode (`splatGaussians`)
- Used when `topK == 0` in the header
- Direct accumulation of all Gaussian contributions
- Formula: `color += gaussian_color * exp(-sigma)`
- Matches the original CUDA `nd_rasterize_forward` kernel

#### 2. TopK Mode (`splatGaussiansTopK`)
- Used when `topK > 0` in the header
- Two-pass algorithm:
  - **Pass 1**: Collect top-K Gaussians based on alpha values
  - **Pass 2**: Accumulate normalized contributions
- Formula: `color += gaussian_color * (alpha_k / sum_topk_alphas)`
- Matches the CUDA `nd_rasterize_forward_topk_norm` kernel

### Algorithm (TopK Mode)

```metal
// Pass 1: Find top-K Gaussians
for each gaussian in tile:
    compute sigma and alpha = exp(-sigma)
    if alpha > min(topk_values):
        replace minimum in topk arrays with this gaussian

// Pass 2: Accumulate with normalization
sum_val = sum(topk_values)
for each k in topk:
    weight = topk_values[k] / (sum_val + EPS)
    color += colors[topk_ids[k]] * weight
```

### File Format Integration

The TopK parameter is stored in the Image-GS binary format header:
- **Offset**: 16 bytes into header
- **Type**: `UInt16`
- **Values**: 
  - `0` = No TopK normalization (use basic mode)
  - `1-10` = Use TopK normalization with K Gaussians

### Swift Integration

The renderer automatically selects the correct compute pipeline based on the header:

```swift
let pipeline = scene.header.topK > 0 ? computePipelineTopK : computePipeline
```

## Performance Considerations

### Basic Mode
- **Pros**: Faster, simpler computation
- **Cons**: May have numerical precision issues with many overlapping Gaussians
- **Use Case**: Models trained without TopK normalization

### TopK Mode
- **Pros**: Better quality, more numerically stable
- **Cons**: Requires two passes over Gaussians, uses more registers
- **Use Case**: Models trained with TopK normalization enabled

### Memory Usage
Each thread in TopK mode uses:
- `10 * sizeof(int)` = 40 bytes for Gaussian IDs
- `10 * sizeof(float)` = 40 bytes for alpha values
- Total: 80 bytes of thread-local memory

## Testing

To verify the implementation is working correctly:

1. **Check Header**: Examine the exported `.igs2` file header to see the `topK` value
2. **Visual Comparison**: Compare Metal rendering with CUDA/PyTorch rendering
3. **Train Models**: Train models with both `--disable-topk-norm` and without

## Compatibility

- **Backward Compatible**: Files with `topK=0` use the basic mode
- **Forward Compatible**: Supports all TopK values from 1-10
- **CUDA Parity**: Matches behavior of CUDA kernels exactly

## References

- Python implementation: `model.py` lines 389-409
- CUDA kernel: `gsplat/gsplat/cuda/csrc/forward.cu` lines 171-393
- File format: `utils/gaussian_io.py` lines 44-115

