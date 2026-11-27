"""Methods for working with MatPES data downloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from monty.io import zopen

from .parquet import is_parquet, iter_rows, row_to_record

REPO_ID = "materialyze/matpes"

PARQUET_VERSIONS = ("2025.3",)
"""Dataset versions distributed as Parquet on Hugging Face. All versions are also available as JSONL."""


def _load_jsonl(path: str) -> list[dict]:
    """Load a MatPES ``.jsonl`` file (one JSON record per line).

    ``zopen`` transparently handles both plain and gzip-compressed files based on the extension.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of record dicts, one per non-empty line.
    """
    with zopen(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_filename(
    functional: str = "PBE",
    version: str = "2025.3",
    split: Literal["train", "valid", "test"] | None = None,
    atoms: bool = False,
    fmt: Literal["jsonl", "parquet"] = "parquet",
) -> str:
    """Return the Hugging Face file name of a MatPES dataset file.

    With ``fmt="parquet"`` (default), versions in :data:`PARQUET_VERSIONS` and the atomic reference energies resolve
    to Parquet files, and earlier versions quietly fall back to their JSONL files. ``fmt="jsonl"`` always resolves to
    JSONL.

    Args:
        functional: "PBE" or "R2SCAN" (case-insensitive).
        version: Dataset version. Ignored for ``atoms``.
        split: Optional 90/5/5 split ("train", "valid" or "test"), available from 2025.2 on.
        atoms: Return the name of the isolated-atom reference energy file instead.
        fmt: Preferred file format, "parquet" (default) or "jsonl".

    Returns:
        File name, e.g., ``MatPES-PBE-2025.3-train.parquet`` or ``MatPES-PBE-2025.2.jsonl``.
    """
    if fmt not in ("jsonl", "parquet"):
        raise ValueError(f"Unknown format {fmt!r}. Use 'parquet' or 'jsonl'.")
    stem = f"MatPES-{functional.upper()}-atoms" if atoms else f"MatPES-{functional.upper()}-{version}"
    if split is not None and not atoms:
        stem += f"-{split}"
    use_parquet = fmt == "parquet" and (atoms or version in PARQUET_VERSIONS)
    return f"{stem}.{'parquet' if use_parquet else 'jsonl'}"


def download(
    functional: str = "PBE",
    version: str = "2025.3",
    split: Literal["train", "valid", "test"] | None = None,
    atoms: bool = False,
    fmt: Literal["jsonl", "parquet"] = "parquet",
    local_dir: str | Path | None = None,
) -> Path:
    """Download a MatPES file from Hugging Face (cached) and return its local path.

    Args:
        functional: "PBE" or "R2SCAN".
        version: Dataset version.
        split: Optional "train", "valid" or "test" split.
        atoms: Download the isolated-atom reference energies instead.
        fmt: "parquet" (default; quietly falls back to JSONL for versions without Parquet files) or "jsonl".
        local_dir: Optional directory to place the file in. Defaults to the Hugging Face cache.

    Returns:
        Local path of the file.
    """
    filename = get_filename(functional, version, split=split, atoms=atoms, fmt=fmt)
    return Path(hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset", local_dir=local_dir))


def load_records(path: str | Path) -> list[dict]:
    """Load a MatPES Parquet or JSONL file as a list of record dicts with a serialized pymatgen ``structure``.

    Args:
        path: Parquet or (optionally gzipped) JSONL file.

    Returns:
        List of record dicts. For Parquet, see :func:`matpes.parquet.row_to_record` for the restored fields.
    """
    if is_parquet(path):
        return [row_to_record(row) for row in iter_rows(path)]
    return _load_jsonl(str(path))


def get_data(
    functional: Literal["PBE", "R2SCAN"] = "PBE",
    version: str = "2025.3",
    return_data: bool = True,
    download_atoms: bool = False,
    fmt: Literal["jsonl", "parquet"] = "parquet",
    split: Literal["train", "valid", "test"] | None = None,
) -> tuple[list[dict], list[dict]] | list[dict] | None:
    """Download and load a MatPES dataset as a list of record dicts, each with a serialized pymatgen ``structure``.

    The return value has the same layout as in matpes <= 0.0.3 whichever file format is downloaded:

    - ``fmt="parquet"`` (default): downloads the Parquet file and rebuilds the records from it (see
      :func:`matpes.parquet.row_to_record`; derived fields such as ``symmetry`` are not regenerated). Versions without
      Parquet files quietly fall back to JSONL.
    - ``fmt="jsonl"``: downloads the JSONL file and returns its records verbatim.

    For columnar access, which is much faster and lighter on memory, use the Parquet file directly, e.g.,
    ``matpes.parquet.read_table(download(...))`` or the helpers in :mod:`matpes.convert`.

    Args:
        functional: "PBE" or "R2SCAN". Defaults to "PBE".
        version: Dataset version. Defaults to "2025.3".
        return_data: Whether to load and return the data. If False, the files are only downloaded.
        download_atoms: Whether to also download and return the isolated-atom reference energies.
        fmt: "parquet" (default) or "jsonl".
        split: Optional "train", "valid" or "test" split instead of the full dataset.

    Returns:
        The dataset, or a tuple of (dataset, atomic reference energies) if ``download_atoms`` is True. None if
        ``return_data`` is False.
    """
    if fmt not in ("jsonl", "parquet"):
        raise ValueError(f"Unknown format {fmt!r}. Use 'parquet' or 'jsonl'.")
    data_path = download(functional, version, split=split, fmt=fmt)
    atoms_path = download(functional, atoms=True, fmt=fmt) if download_atoms else None

    if not return_data:
        return None

    data = load_records(data_path)
    if atoms_path is None:
        return data
    return data, load_records(atoms_path)


def get_atom_energies(functional: Literal["PBE", "R2SCAN"] = "PBE") -> dict[str, float]:
    """Return the isolated-atom reference energies (eV) computed with MatPES settings, keyed by element symbol.

    Args:
        functional: "PBE" or "R2SCAN".

    Returns:
        Dict of element symbol to energy.
    """
    table = pq.read_table(download(functional, atoms=True), columns=["species", "energy"])
    return {species[0]: energy for species, energy in zip(*table.to_pydict().values(), strict=True)}
