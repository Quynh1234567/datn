"""Số học điểm cố định Q(m.n) bù hai – dùng chung cho Golden Model và bộ xuất .mem."""
from __future__ import annotations

import math

import numpy as np


def qrange(width: int, signed: bool = True) -> tuple[int, int]:
    return (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)


def to_fixed(x, width: int, frac: int, signed: bool = True, return_sat: bool = False):
    """Làm tròn gần nhất + bão hòa về [min, max] của số `width` bit, `frac` bit lẻ."""
    lo, hi = qrange(width, signed)
    q = np.round(np.asarray(x, np.float64) * (1 << frac)).astype(np.int64)
    sat = int(np.count_nonzero((q < lo) | (q > hi)))
    q = np.clip(q, lo, hi)
    return (q, sat) if return_sat else q


def from_fixed(q, frac: int) -> np.ndarray:
    return np.asarray(q, np.float64) / (1 << frac)


def to_hex(values, width: int) -> list[str]:
    """Số nguyên (có thể âm) → chuỗi hex bù hai đủ ⌈width/4⌉ ký tự, dùng cho $readmemh."""
    mask = (1 << width) - 1
    nd = math.ceil(width / 4)
    return [format(int(v) & mask, f"0{nd}X") for v in np.asarray(values).ravel()]


def to_bin(values, width: int) -> list[str]:
    mask = (1 << width) - 1
    return [format(int(v) & mask, f"0{width}b") for v in np.asarray(values).ravel()]


def bits_needed(max_abs: float, signed: bool = True) -> int:
    if max_abs <= 0:
        return 1
    return int(math.ceil(math.log2(max_abs + 1))) + (1 if signed else 0)
