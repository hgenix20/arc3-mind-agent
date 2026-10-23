"""Screen images for the vision model: PNG bytes from a grid, with no imaging library.

Contract:
- ``png`` draws a 64x64 grid as an 8-bit palette PNG, each cell ``scale`` pixels square.
  Index = ARC color 0..15; padding (-1) is drawn black (palette index 16).
- ``diff_png`` draws the ``after`` frame as an RGB PNG with unchanged cells darkened to
  ``DIM_PERCENT`` percent and changed cells at full color. At ``scale >= BORDER_MIN_SCALE``
  each changed cell also gets a 1-pixel white border.
- ``data_url`` wraps PNG bytes as a ``data:image/png;base64,...`` URL.
- Pure functions; no I/O. A grid of the wrong rank raises ``ValueError``.
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np
import numpy.typing as npt

from arcagent.core.types import Grid

_HEX = [
    "000000",
    "0074D9",
    "FF4136",
    "2ECC40",
    "FFDC00",
    "AAAAAA",
    "F012BE",
    "FF851B",
    "7FDBFF",
    "870C25",
    "FFFFFF",
    "3D9970",
    "B10DC9",
    "39CCCC",
    "85144B",
    "F7F7F7",
]
PALETTE: tuple[tuple[int, int, int], ...] = tuple(
    (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) for h in _HEX
)
SCALE = 8
DIM_PERCENT = 30
BORDER_MIN_SCALE = 4
_PAD_INDEX = len(PALETTE)  # palette entry 16: padding, black

_Bytes2D = npt.NDArray[np.uint8]


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def _encode(pixels: _Bytes2D, *, width: int, height: int, color_type: int, plte: bytes) -> bytes:
    """``pixels`` is (height, row_bytes) uint8; each row gets filter type 0."""
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), pixels], axis=1).tobytes()
    out = b"\x89PNG\r\n\x1a\n"
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    if plte:
        out += _chunk(b"PLTE", plte)
    out += _chunk(b"IDAT", zlib.compress(raw, 6))
    return out + _chunk(b"IEND", b"")


def _check(grid: Grid, scale: int) -> None:
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2-D, got {grid.ndim}-D")
    if scale < 1:
        raise ValueError(f"scale must be at least 1, got {scale}")


def _upscale(a: npt.NDArray[np.generic], scale: int) -> npt.NDArray[np.generic]:
    return np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)


def png(grid: Grid, *, scale: int = SCALE) -> bytes:
    """Palette PNG of ``grid``; padding (-1) and any out-of-range value draw black."""
    _check(grid, scale)
    idx = np.where((grid >= 0) & (grid < len(PALETTE)), grid, _PAD_INDEX).astype(np.uint8)
    big: _Bytes2D = _upscale(idx, scale).astype(np.uint8)
    colors = [*PALETTE, (0, 0, 0)]
    plte = bytes(v for rgb in colors for v in rgb)
    h, w = big.shape
    return _encode(big, width=w, height=h, color_type=3, plte=plte)


def _rgb(grid: Grid) -> npt.NDArray[np.uint8]:
    table = np.array([*PALETTE, (0, 0, 0)], dtype=np.uint8)
    idx = np.where((grid >= 0) & (grid < len(PALETTE)), grid, _PAD_INDEX).astype(np.intp)
    return table[idx]


def diff_png(before: Grid, after: Grid, *, scale: int = SCALE) -> bytes:
    """RGB PNG of ``after`` with unchanged cells dimmed and changed cells highlighted."""
    _check(before, scale)
    _check(after, scale)
    if before.shape != after.shape:
        raise ValueError(f"shape mismatch: {before.shape} vs {after.shape}")
    changed = before != after
    rgb = _rgb(after).astype(np.uint16)
    rgb = np.where(changed[:, :, None], rgb, rgb * DIM_PERCENT // 100).astype(np.uint8)
    big = _upscale(rgb, scale).astype(np.uint8)
    if scale >= BORDER_MIN_SCALE:
        edge = np.zeros((scale, scale), dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        ring = np.tile(edge, changed.shape) & _upscale(changed, scale).astype(bool)
        big[ring] = (255, 255, 255)
    h, w = big.shape[:2]
    return _encode(big.reshape(h, w * 3), width=w, height=h, color_type=2, plte=b"")


def data_url(png_bytes: bytes) -> str:
    """The PNG as a base64 ``data:`` URL."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
