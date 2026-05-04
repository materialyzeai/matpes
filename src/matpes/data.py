"""Methods for working with MatPES data downloads."""

from __future__ import annotations

import gzip
import json
from typing import Literal

from huggingface_hub import hf_hub_download

REPO_ID = "materialyze/matpes"


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
    entries unless otherwise configured.

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
        repo_id=REPO_ID, filename=f"MatPES-{functional.upper()}-{version}.json", repo_type="dataset"
    )
    atoms_path = ""
    if download_atoms:
        atoms_path = hf_hub_download(
            repo_id=REPO_ID, filename=f"MatPES-{functional.upper()}-atoms.json", repo_type="dataset"
        )

    if not return_data:
        return None

    with gzip.open(data_path, "rt") as f:
        data = json.load(f)

    if download_atoms:
        with gzip.open(atoms_path, "rt") as f:
            atoms_data = json.load(f)

        return data, atoms_data

    return data
