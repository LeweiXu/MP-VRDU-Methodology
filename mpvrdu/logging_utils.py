"""Structured logging + seed control (Stage 0)."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import random
import sys
from typing import Iterator

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _formatter() -> logging.Formatter:
    return logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)


def _file_handler(path: str | Path, mode: str = "a") -> logging.FileHandler:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    handler.setFormatter(_formatter())
    return handler


def get_logger(name: str = "mpvrdu", level: int | None = None) -> logging.Logger:
    """Return a process-wide logger with a consistent format.

    Level can be overridden by the MPVRDU_LOGLEVEL env var (e.g. DEBUG).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        env_level = os.environ.get("MPVRDU_LOGLEVEL", "INFO").upper()
        root_level = level if level is not None else getattr(logging, env_level, logging.INFO)
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(_formatter())
        root = logging.getLogger("mpvrdu")
        root.handlers.clear()
        root.addHandler(handler)
        if os.environ.get("MPVRDU_LOG_FILE"):
            root.addHandler(_file_handler(os.environ["MPVRDU_LOG_FILE"]))
        root.setLevel(root_level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("mpvrdu") else f"mpvrdu.{name}")


def add_file_logging(path: str | Path, mode: str = "a") -> logging.Handler:
    """Attach a persistent log file to the process-wide MP-VRDU logger."""
    get_logger()
    handler = _file_handler(path, mode=mode)
    logging.getLogger("mpvrdu").addHandler(handler)
    return handler


@contextmanager
def file_logging(path: str | Path, mode: str = "a") -> Iterator[Path]:
    """Temporarily duplicate all MP-VRDU log records to ``path``."""
    handler = add_file_logging(path, mode=mode)
    try:
        yield Path(path)
    finally:
        root = logging.getLogger("mpvrdu")
        root.removeHandler(handler)
        handler.close()


def set_seed(seed: int) -> None:
    """Fix RNG seeds for reproducibility. Torch/numpy seeded if importable."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
