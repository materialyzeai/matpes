"""Methods for working with MatPES data downloads."""

from __future__ import annotations

import json
from typing import Literal

from huggingface_hub import hf_hub_download
from monty.io import zopen

REPO_ID = "materialyze/matpes"


def _load_jsonl(path: str) -> list[dict]:
    """Load a MatPES ``.jsonl`` file (one JSON record per line).

    MatPES datasets are distributed as JSONL rather than a single JSON array so
    that records can be streamed one line at a time instead of parsing the whole
    file into memory at once. ``zopen`` transparently handles both plain and
    gzip-compressed files based on the extension.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of record dicts, one per non-empty line.
    """
    with zopen(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_data(
    functional: Literal["PBE", "R2SCAN"] = "PBE",
    version: str = "2025.2",
    return_data: bool = True,
    download_atoms: bool = False,
) -> tuple[list[dict], list[dict]] | list[dict] | None:
    """
    Retrieves dataset(s) related to materials properties based on specified options.

    This function loads a dataset corresponding to a given functional and optionally
    downloads additional atomic data. It allows specifying the functional type
    (e.g., "PBE" or "R2SCAN") and the dataset version. By default, the output includes
    entries unless otherwise configured. MatPES datasets are distributed as JSONL
    files (one JSON record per line), which are read line by line via
    :func:`_load_jsonl` rather than loaded as a single JSON array.

    Parameters:
        functional (Literal["PBE", "R2SCAN"]): The functional type specifying the
            dataset to retrieve. Defaults to "PBE".
        version (str): The version of the dataset to retrieve. Defaults to "2025.2".
        download_atoms (bool): Whether to download and include atomic data in
            the output. Defaults to False.

    Return Values:
        Either the primary dataset or both the primary dataset and atomic data
        depending on the value of `download_atoms`. If `download_atoms` is False, it
        returns the primary dataset. Otherwise, it returns a tuple containing the
        primary dataset and atomic data.

    Exceptions:
        None
    """
    data_path = hf_hub_download(
        repo_id=REPO_ID, filename=f"MatPES-{functional.upper()}-{version}.jsonl", repo_type="dataset"
    )
    atoms_path = ""
    if download_atoms:
        atoms_path = hf_hub_download(
            repo_id=REPO_ID, filename=f"MatPES-{functional.upper()}-atoms.jsonl", repo_type="dataset"
        )

    if not return_data:
        return None

    data = _load_jsonl(data_path)

    if download_atoms:
        atoms_data = _load_jsonl(atoms_path)
        return data, atoms_data

    return data
