#!/usr/bin/env python3
"""Dependency-free perceptual hash (pHash) — 64-bit, DCT-based.

Shared by `precull_frames.py` (STO-SCN-092) and `camera_viewer/clustering.py`.
Replaces the `imagehash` dependency (which pulled in PyWavelets) with a
self-contained numpy + PIL implementation so the pre-cull stage "runs anywhere"
with no extra install (T-014).

Algorithm (the classic DCT pHash):
    grayscale -> 32x32 -> 2D DCT-II -> top-left 8x8 -> median threshold -> 64 bits

The DCT here is unnormalized DCT-II via a small cosine matrix; only the *sign*
relative to the median matters for the hash bits, so any positive scaling of the
transform yields the same fingerprint as the scipy/imagehash variant.
"""
from __future__ import annotations

import functools

import numpy as np
from PIL import Image

HASH_SIZE = 8          # low-frequency block kept (8x8 = 64 bits)
_IMG = HASH_SIZE * 4   # 32x32 working image (classic highfreq_factor=4)


@functools.lru_cache(maxsize=4)
def _dct_matrix(n: int) -> np.ndarray:
    """Unnormalized DCT-II basis matrix M[k, i] = cos(pi*(2i+1)*k / (2n))."""
    i = np.arange(n)
    k = i.reshape(-1, 1)
    return np.cos(np.pi * (2 * i + 1) * k / (2 * n))


def _dct2(a: np.ndarray) -> np.ndarray:
    m = _dct_matrix(a.shape[0])
    return m @ a @ m.T


def _to_gray32(img) -> np.ndarray:
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    im = im.convert("L").resize((_IMG, _IMG), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def phash(img) -> np.uint64:
    """64-bit perceptual hash of a PIL image or (H,W[,3]) uint8 array."""
    block = _dct2(_to_gray32(img))[:HASH_SIZE, :HASH_SIZE]
    med = np.median(block)
    bits = (block > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    return np.uint64(val)


def phash_file(path) -> np.uint64:
    with Image.open(path) as im:
        return phash(im)


def hamming(a, b) -> int:
    """Hamming distance between two 64-bit hashes (0..64)."""
    return int(bin(int(a) ^ int(b)).count("1"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        print(int(phash_file(sys.argv[1])))
    elif len(sys.argv) == 3:
        print(hamming(phash_file(sys.argv[1]), phash_file(sys.argv[2])))
    else:
        sys.exit("usage: phash.py <img> [<img2>]   (hash, or hamming of two)")
