# Image-GS 2D Gaussian Splat Format (IGS2)

This document describes the portable binary container used to exchange 2D Gaussian splat scenes between the Python training pipeline and the Metal renderer.

## File Overview

An `.igs2` file is divided into a fixed-size header followed by a sequence of typed data chunks. Numeric data is stored in little-endian order.

```
+------------------+--------------------+
| 64-byte header   | variable chunks... |
+------------------+--------------------+
```

### Header Layout

| Offset | Size | Field              | Description |
| ------ | ---- | ------------------ | ----------- |
| 0      | 4    | `magic`            | ASCII `IGS2` signature |
| 4      | 1    | `version_major`    | Major version (currently 0) |
| 5      | 1    | `version_minor`    | Minor version (currently 1) |
| 6      | 1    | `flags`            | Reserved feature flags (0) |
| 7      | 1    | `padding`          | Reserved (0) |
| 8      | 2    | `tile_width`       | Tile width used during binning (pixels) |
| 10     | 2    | `tile_height`      | Tile height used during binning (pixels) |
| 12     | 2    | `channels`         | Number of per-Gaussian feature channels |
| 14     | 2    | `topk`             | Top-k value for normalization (0 when disabled) |
| 16     | 4    | `image_width`      | Raster width in pixels |
| 20     | 4    | `image_height`     | Raster height in pixels |
| 24     | 4    | `gaussian_count`   | Number of stored Gaussian primitives |
| 28     | 4    | `intersection_count` | Total number of tile/gaussian intersections |
| 32     | 32   | `reserved`         | Reserved for future use (zero filled) |

### Chunk Layout

Each chunk begins with a 12-byte header followed by the raw payload and an explicit shape descriptor:

| Offset | Size | Field        | Description |
| ------ | ---- | ------------ | ----------- |
| 0      | 4    | `tag`        | Four-character ASCII identifier |
| 4      | 2    | `dtype_code` | Numeric dtype enum (see below) |
| 6      | 2    | `rank`       | Number of dimensions |
| 8      | 4    | `byte_length`| Size of the payload in bytes |
| 12     | 4×`rank` | `shape`  | Unsigned int32 extents (row-major) |
| ...    | `byte_length` | `payload` | Raw array data |

Supported `dtype_code` values:

| Code | NumPy dtype |
| ---- | ----------- |
| 0    | `float16` |
| 1    | `float32` |
| 2    | `float64` |
| 3    | `uint16` |
| 4    | `uint32` |
| 5    | `int16` |
| 6    | `int32` |

### Standard Chunks

The current renderer expects the following chunks (stored in standard row-major order):

| Tag  | Shape             | Dtype    | Description |
| ---- | ----------------- | -------- | ----------- |
| `CNTR` | `(N, 2)`          | `float32` | Gaussian centers in raster pixel coordinates |
| `CNIC` | `(N, 3)`          | `float32` | Upper triangular conic coefficients `(a, b, c)` |
| `COLR` | `(N, C)`          | `float32` | Per-Gaussian feature/color values |
| `GIDX` | `(M,)`            | `uint32`  | Gaussian indices sorted by tile |
| `TBIN` | `(T, 2)`          | `uint32`  | Tile ranges `start, end` into `GIDX`

Here `N` equals `gaussian_count`, `M` equals `intersection_count`, and `T` is the number of tiles (`tile_width` × `tile_height`).

## Export Path

The `GaussianSplatting2D.export_gaussians(path)` helper packages the in-memory model state into this format. The exporter reuses the project-and-bin pipeline from the CUDA renderer, preserving the exact tiling configuration for fast Metal inference.

## Extensibility

Additional metadata or per-Gaussian attributes can be introduced by appending new chunks. Readers should ignore tags they do not recognize.
