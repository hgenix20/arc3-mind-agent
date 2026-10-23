"""PNG writers: signature, size, palette round trip, diff highlighting."""

from __future__ import annotations

import struct
import zlib

import numpy as np

from arcagent.mind.screen import PALETTE, data_url, diff_png, png

RGB = tuple[int, int, int]


def _chunks(b: bytes) -> dict[bytes, bytes]:
    out: dict[bytes, bytes] = {}
    pos = 8
    while pos < len(b):
        (n,) = struct.unpack(">I", b[pos : pos + 4])
        tag = b[pos + 4 : pos + 8]
        data = b[pos + 8 : pos + 8 + n]
        (crc,) = struct.unpack(">I", b[pos + 8 + n : pos + 12 + n])
        assert crc == zlib.crc32(tag + data)
        out[tag] = out.get(tag, b"") + data
        pos += 12 + n
    return out


def _rows(b: bytes, bpp: int) -> tuple[list[bytes], dict[bytes, bytes]]:
    ch = _chunks(b)
    w, h = struct.unpack(">II", ch[b"IHDR"][:8])
    raw = zlib.decompress(ch[b"IDAT"])
    stride = 1 + w * bpp
    rows: list[bytes] = []
    for y in range(h):
        assert raw[y * stride] == 0
        rows.append(raw[y * stride + 1 : (y + 1) * stride])
    return rows, ch


def decode_palette_png(b: bytes) -> list[list[RGB]]:
    rows, ch = _rows(b, 1)
    plte = ch[b"PLTE"]
    return [[(plte[3 * i], plte[3 * i + 1], plte[3 * i + 2]) for i in r] for r in rows]


def decode_rgb_png(b: bytes) -> list[list[RGB]]:
    rows, _ = _rows(b, 3)
    return [[(r[i], r[i + 1], r[i + 2]) for i in range(0, len(r), 3)] for r in rows]


def test_png_signature_and_size() -> None:
    b = png(np.zeros((64, 64), dtype=np.int8))
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", b[16:24])
    assert (w, h) == (512, 512)


def test_png_decodes_back_to_palette_indices() -> None:
    g = np.zeros((64, 64), dtype=np.int8)
    g[3, 5] = 9
    g[0, 0] = -1
    rows = decode_palette_png(png(g, scale=1))
    assert rows[3][5] == PALETTE[9] and rows[0][0] == (0, 0, 0)


def test_png_upscale_fills_the_block() -> None:
    g = np.zeros((64, 64), dtype=np.int8)
    g[1, 2] = 4
    rows = decode_palette_png(png(g))
    assert rows[8][16] == PALETTE[4] and rows[15][23] == PALETTE[4]
    assert rows[8][24] == PALETTE[0] and rows[16][16] == PALETTE[0]


def test_diff_png_darkens_unchanged_and_keeps_changed() -> None:
    a = np.zeros((64, 64), dtype=np.int8)
    b = a.copy()
    b[10, 10] = 4
    b[20, 20] = 2
    a[20, 20] = 2
    rows = decode_rgb_png(diff_png(a, b, scale=1))
    assert rows[10][10] == PALETTE[4] and rows[0][0] == (0, 0, 0)
    assert rows[20][20] == tuple(v * 30 // 100 for v in PALETTE[2])


def test_diff_png_borders_changed_cells_at_scale() -> None:
    a = np.zeros((64, 64), dtype=np.int8)
    b = a.copy()
    b[2, 3] = 4
    rows = decode_rgb_png(diff_png(a, b))
    assert rows[16][24] == (255, 255, 255) and rows[23][31] == (255, 255, 255)
    assert rows[19][27] == PALETTE[4]
    assert rows[15][24] == (0, 0, 0)


def test_data_url_prefix() -> None:
    assert data_url(b"x").startswith("data:image/png;base64,")
