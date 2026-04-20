"""Tools for working with MatPES."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matpes")
except PackageNotFoundError:
    pass  # package not installed

MATPES_SRC = "https://huggingface.co/datasets/Materialyze/matpes/resolve/main/"
