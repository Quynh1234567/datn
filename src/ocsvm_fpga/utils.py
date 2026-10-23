"""Tiện ích chung: logging, thanh tiến trình, seed, lưu JSON."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")

try:  # tqdm là tùy chọn
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


def get_logger(name: str = "ocsvm_fpga", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
            logger.addHandler(fh)
    return logger


def progress(it: Iterable[T], desc: str = "", total: int | None = None) -> Iterator[T]:
    if _tqdm is not None:
        yield from _tqdm(it, desc=desc, total=total, ncols=90)
        return
    total = total if total is not None else (len(it) if hasattr(it, "__len__") else None)
    t0 = time.time()
    for i, x in enumerate(it, 1):
        yield x
        if total and (i == total or i % max(1, total // 20) == 0):
            print(f"\r{desc}: {i}/{total} ({time.time() - t0:.0f}s)", end="", flush=True)
    if total:
        print()


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def save_json(obj, path: str | Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer(logger: logging.Logger, msg: str):
    t0 = time.time()
    yield
    logger.info(f"{msg} – {time.time() - t0:.1f}s")
