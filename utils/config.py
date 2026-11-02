"""
utils/config.py
───────────────
Loads config.yaml into a nested SimpleNamespace so values are
accessible with attribute notation: cfg.training.num_rounds

Usage
-----
    from utils.config import load_config
    cfg = load_config()          # reads config.yaml from cwd
    cfg = load_config("path/to/config.yaml")
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def _dict_to_ns(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict to SimpleNamespace objects."""
    ns = SimpleNamespace()
    for key, value in d.items():
        setattr(ns, key, _dict_to_ns(value) if isinstance(value, dict) else value)
    return ns


def load_config(path: str | Path = "config.yaml") -> SimpleNamespace:
    """
    Parse a YAML config file and return a nested SimpleNamespace.

    Raises FileNotFoundError if the path does not exist so the error
    is clear rather than an AttributeError later.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}\n"
            "Make sure you are running from the project root."
        )
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)
