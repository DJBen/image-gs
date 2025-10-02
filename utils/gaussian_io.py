"""Utilities for serializing 2D Gaussian splat data to the Image-GS portable format."""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Tuple

import numpy as np

MAGIC = b"IGS2"
VERSION_MAJOR = 0
VERSION_MINOR = 1
HEADER_SIZE = 64  # bytes

# Mapping between numpy dtypes and compact integer codes so the file can be parsed
DTYPE_CODES: Dict[np.dtype, int] = {
    np.dtype(np.float16): 0,
    np.dtype(np.float32): 1,
    np.dtype(np.float64): 2,
    np.dtype(np.uint16): 3,
    np.dtype(np.uint32): 4,
    np.dtype(np.int16): 5,
    np.dtype(np.int32): 6,
}
CODE_TO_DTYPE = {code: dtype for dtype, code in DTYPE_CODES.items()}
CHUNK_HEADER_STRUCT = struct.Struct("<4sHHi")  # tag, dtype code, rank, byte length


def _dtype_code(dtype: np.dtype) -> int:
    dtype = np.dtype(dtype)
    if dtype not in DTYPE_CODES:
        raise ValueError(f"Unsupported dtype '{dtype}' for gaussian format")
    return DTYPE_CODES[dtype]


def _ensure_le(array: np.ndarray) -> np.ndarray:
    if array.dtype.byteorder in ("<", "="):
        return array
    return array.byteswap().newbyteorder()


@dataclass
class GaussianSplatHeader:
    """Header metadata for an Image-GS 2D gaussian splat file."""

    image_width: int
    image_height: int
    tile_width: int
    tile_height: int
    channels: int
    gaussian_count: int
    intersection_count: int
    topk: int = 0
    flags: int = 0
    reserved: bytes = field(default=b"\x00" * (HEADER_SIZE - 32))

    def pack(self) -> bytes:
        if len(self.reserved) != HEADER_SIZE - 32:
            raise ValueError("Reserved portion must keep the header size constant")
        base = struct.pack(
            "<4sBBBBHHHHIII",
            MAGIC,
            VERSION_MAJOR,
            VERSION_MINOR,
            self.flags,
            0,  # padding byte for alignment
            self.tile_width,
            self.tile_height,
            self.channels,
            self.topk,
            self.image_width,
            self.image_height,
            self.gaussian_count,
        )
        base += struct.pack("<I", self.intersection_count)
        return base + self.reserved

    @classmethod
    def unpack(cls, payload: bytes) -> "GaussianSplatHeader":
        if len(payload) != HEADER_SIZE:
            raise ValueError("Invalid header length")
        head = payload[:32]
        (
            magic,
            v_major,
            v_minor,
            flags,
            _pad,
            tile_w,
            tile_h,
            channels,
            topk,
            width,
            height,
            gaussians,
        ) = struct.unpack("<4sBBBBHHHHIII", head[:28])
        if magic != MAGIC:
            raise ValueError("Unrecognized magic header")
        if v_major != VERSION_MAJOR:
            raise ValueError(f"Unsupported major version {v_major}.{v_minor}")
        (intersections,) = struct.unpack("<I", head[28:32])
        reserved = payload[32:]
        return cls(
            image_width=width,
            image_height=height,
            tile_width=tile_w,
            tile_height=tile_h,
            channels=channels,
            gaussian_count=gaussians,
            intersection_count=intersections,
            topk=topk,
            flags=flags,
            reserved=reserved,
        )


@dataclass
class GaussianBuffer:
    centers: np.ndarray  # shape (N, 2), float32
    conics: np.ndarray   # shape (N, 3), float32
    colors: np.ndarray   # shape (N, C), float32
    gaussian_ids_sorted: np.ndarray  # shape (M,), uint32
    tile_bins: np.ndarray  # shape (T, 2), uint32


CHUNK_TAGS = {
    "centers": b"CNTR",
    "conics": b"CNIC",
    "colors": b"COLR",
    "gaussian_ids_sorted": b"GIDX",
    "tile_bins": b"TBIN",
}


def _write_chunk(fh: BinaryIO, tag: bytes, array: np.ndarray) -> None:
    array = _ensure_le(np.asarray(array))
    data = array.tobytes(order="C")
    header = CHUNK_HEADER_STRUCT.pack(tag, _dtype_code(array.dtype), array.ndim, len(data))
    fh.write(header)
    fh.write(struct.pack(f"<{array.ndim}I", *array.shape))
    fh.write(data)


def save_gaussian_splats(path: str, header: GaussianSplatHeader, buffers: GaussianBuffer) -> None:
    """Serialize gaussian splat data to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if buffers.centers.shape[0] != header.gaussian_count:
        raise ValueError("centers count mismatch")
    if buffers.conics.shape[0] != header.gaussian_count:
        raise ValueError("conics count mismatch")
    if buffers.colors.shape[0] != header.gaussian_count:
        raise ValueError("colors count mismatch")
    if header.intersection_count != buffers.gaussian_ids_sorted.size:
        raise ValueError("intersection count mismatch")

    with open(path, "wb") as fh:
        fh.write(header.pack())
        _write_chunk(fh, CHUNK_TAGS["centers"], buffers.centers.astype(np.float32, copy=False))
        _write_chunk(fh, CHUNK_TAGS["conics"], buffers.conics.astype(np.float32, copy=False))
        _write_chunk(fh, CHUNK_TAGS["colors"], buffers.colors.astype(np.float32, copy=False))
        _write_chunk(fh, CHUNK_TAGS["gaussian_ids_sorted"], buffers.gaussian_ids_sorted.astype(np.uint32, copy=False))
        _write_chunk(fh, CHUNK_TAGS["tile_bins"], buffers.tile_bins.astype(np.uint32, copy=False))


def load_gaussian_splats(path: str) -> Tuple[GaussianSplatHeader, GaussianBuffer]:
    """Load gaussian splat data from disk."""
    with open(path, "rb") as fh:
        header_blob = fh.read(HEADER_SIZE)
        header = GaussianSplatHeader.unpack(header_blob)
        arrays: Dict[bytes, np.ndarray] = {}
        while True:
            chunk_header = fh.read(CHUNK_HEADER_STRUCT.size)
            if not chunk_header:
                break
            tag, dtype_code, rank, byte_len = CHUNK_HEADER_STRUCT.unpack(chunk_header)
            shape = struct.unpack(f"<{rank}I", fh.read(4 * rank))
            payload = fh.read(byte_len)
            dtype = CODE_TO_DTYPE.get(dtype_code)
            if dtype is None:
                raise ValueError(f"Unknown dtype code {dtype_code}")
            array = np.frombuffer(payload, dtype=dtype).reshape(shape)
            arrays[tag] = array
    try:
        centers = arrays[CHUNK_TAGS["centers"]]
        conics = arrays[CHUNK_TAGS["conics"]]
        colors = arrays[CHUNK_TAGS["colors"]]
        gidx = arrays[CHUNK_TAGS["gaussian_ids_sorted"]]
        tbins = arrays[CHUNK_TAGS["tile_bins"]]
    except KeyError as exc:
        missing = exc.args[0].decode("ascii") if isinstance(exc.args[0], bytes) else str(exc)
        raise KeyError(f"Gaussian splat file missing chunk '{missing}'") from exc
    return header, GaussianBuffer(centers, conics, colors, gidx, tbins)
