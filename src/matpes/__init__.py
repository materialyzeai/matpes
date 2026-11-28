"""Tools for working with MatPES."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matpes")
except PackageNotFoundError:
    pass  # package not installed

MATPES_HF = "https://huggingface.co/datasets/materialyze/matpes"
"""Hugging Face page of the MatPES dataset."""

MATPES_SRC = f"{MATPES_HF}/resolve/main"
"""Base URL for downloading files from the latest MatPES release."""
