"""Nạp và truy cập cấu hình YAML theo kiểu thuộc tính (cfg.model.nu)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """dict cho phép truy cập lồng nhau bằng dấu chấm: cfg.features.feature_set."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(f"Thiếu khóa cấu hình '{key}'") from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _wrap(d: dict) -> Config:
    return Config({k: _wrap(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Đọc configs/default.yaml, gộp file người dùng và các ghi đè dạng 'a.b=value'."""
    with open(PROJECT_ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != (PROJECT_ROOT / "configs" / "default.yaml"):
        with open(path, encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return _wrap(cfg)


def resolve_path(p: str | Path | None) -> Path | None:
    """Đường dẫn tương đối → tính từ thư mục gốc project."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
