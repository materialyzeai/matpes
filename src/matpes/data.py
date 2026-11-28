"""Methods for working with MatPES data downloads.

Layout of the Hugging Face dataset repository (``materialyze/matpes``). Each release is a git tag (``v2025.3``, ...);
``main`` holds only the latest release, as Parquet:

- ``pbe/{train,validation,test}.parquet`` and ``r2scan/{train,validation,test}.parquet``: the dataset as a random
  90/5/5 split. The full dataset is the union of the three files.
- ``atoms/{pbe,r2scan}.parquet``: isolated-atom reference energies.

Releases before 2025.3 are only available as JSONL, with flat file names such as
``MatPES-PBE-2025.2[-train|-valid|-test].jsonl``, at their own tags. :func:`download` resolves the right revision and
files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from monty.io import zopen

from .parquet import is_parquet, iter_rows, row_to_record

REPO_ID = "materialyze/matpes"

LATEST_VERSION = "2025.3"
"""Latest MatPES release."""

PARQUET_VERSIONS = ("2025.3",)
"""Releases distributed as Parquet (directory layout). Earlier releases are JSONL only."""

SPLITS = ("train", "validation", "test")
"""Splits of the Parquet releases, in the order the full dataset is assembled."""

JSONL_VERSIONS = ("2025.1", "2025.2")
"""Releases distributed as JSONL (flat file names, at their own tags)."""

type Split = Literal["train", "valid", "validation", "test"]


def get_revision(version: str = LATEST_VERSION) -> str:
    """Return the git tag of a MatPES release on Hugging Face, e.g., ``v2025.3``."""
    return f"v{version}"


def _normalize_split(split: Split | None) -> str | None:
    if split is None:
        return None
    split = "validation" if split == "valid" else split
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Use one of {SPLITS} (or 'valid').")
    return split


def get_repo_files(
    functional: str = "PBE",
    version: str = LATEST_VERSION,
    split: Split | None = None,
    atoms: bool = False,
    fmt: Literal["jsonl", "parquet"] = "parquet",
) -> tuple[str, list[str]]:
    """Return the Hugging Face revision and file paths that make up a MatPES dataset.

    With ``fmt="parquet"`` (default), releases in :data:`PARQUET_VERSIONS` and the atomic reference energies resolve to
    Parquet files, and earlier releases quietly fall back to their JSONL files. ``fmt="jsonl"`` always resolves to
    JSONL.

    Args:
        functional: "PBE" or "R2SCAN" (case-insensitive).
        version: Release. For ``atoms``, only selects the revision.
        split: Optional "train", "valid"/"validation" or "test" split (available from 2025.2 on). None for the full
            dataset, which for Parquet releases consists of all three split files.
        atoms: Resolve the isolated-atom reference energies instead.
        fmt: Preferred file format, "parquet" (default) or "jsonl" (2025.1 and 2025.2 only).

    Returns:
        Tuple of (revision, list of file paths in the repository).

    Raises:
        ValueError: For an unknown format, split or release.
    """
    if fmt not in ("jsonl", "parquet"):
        raise ValueError(f"Unknown format {fmt!r}. Use 'parquet' or 'jsonl'.")
    name = _normalize_split(split)
    func = functional.lower()
    releases = JSONL_VERSIONS + PARQUET_VERSIONS
    if version not in releases:
        raise ValueError(f"Unknown MatPES release {version!r}. Available: {', '.join(releases)}.")

    if fmt == "parquet" and (atoms or version in PARQUET_VERSIONS):
        revision = get_revision(version if version in PARQUET_VERSIONS else LATEST_VERSION)
        if atoms:
            return revision, [f"atoms/{func}.parquet"]
        return revision, [f"{func}/{s}.parquet" for s in ([name] if name else SPLITS)]

    if version not in JSONL_VERSIONS:
        raise ValueError(f"MatPES {version} is only available as Parquet. Use fmt='parquet'.")
    stem = f"MatPES-{functional.upper()}-atoms" if atoms else f"MatPES-{functional.upper()}-{version}"
    if name is not None and not atoms:
        stem += f"-{'valid' if name == 'validation' else name}"
    return get_revision(version), [f"{stem}.jsonl"]


def download(
    functional: str = "PBE",
    version: str = LATEST_VERSION,
    split: Split | None = None,
    atoms: bool = False,
    fmt: Literal["jsonl", "parquet"] = "parquet",
    local_dir: str | Path | None = None,
) -> Path:
    """Download a MatPES dataset from Hugging Face (cached) and return its local path.

    Args:
        functional: "PBE" or "R2SCAN".
        version: Release.
        split: Optional "train", "valid"/"validation" or "test" split.
        atoms: Download the isolated-atom reference energies instead.
        fmt: "parquet" (default; quietly falls back to JSONL for releases without Parquet files) or "jsonl".
        local_dir: Optional directory to place the files in. Defaults to the Hugging Face cache.

    Returns:
        Local path. For the full dataset of a Parquet release this is the directory holding the train, validation and
        test files, which every reader in :mod:`matpes.parquet` and :mod:`matpes.convert` accepts. Otherwise it is
        the single downloaded file.
    """
    revision, files = get_repo_files(functional, version, split=split, atoms=atoms, fmt=fmt)
    paths = [
        Path(hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", revision=revision, local_dir=local_dir))
        for f in files
    ]
    return paths[0] if len(paths) == 1 else paths[0].parent


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


def load_records(path: str | Path) -> list[dict]:
    """Load a MatPES Parquet file or directory, or a JSONL file, as record dicts with a pymatgen ``structure``.

    Args:
        path: Parquet file, directory of Parquet files, or (optionally gzipped) JSONL file.

    Returns:
        List of record dicts. For Parquet, see :func:`matpes.parquet.row_to_record` for the restored fields.
    """
    if is_parquet(path):
        return [row_to_record(row) for row in iter_rows(path)]
    return _load_jsonl(str(path))


def get_data(
    functional: Literal["PBE", "R2SCAN"] = "PBE",
    version: str = LATEST_VERSION,
    return_data: bool = True,
    download_atoms: bool = False,
    fmt: Literal["jsonl", "parquet"] = "parquet",
    split: Split | None = None,
) -> tuple[list[dict], list[dict]] | list[dict] | None:
    """Download and load a MatPES dataset as a list of record dicts, each with a serialized pymatgen ``structure``.

    The return value has the same layout as in matpes <= 0.0.3 whichever file format is downloaded:

    - ``fmt="parquet"`` (default): downloads the Parquet files and rebuilds the records from them (see
      :func:`matpes.parquet.row_to_record`; derived fields such as ``symmetry`` are not regenerated). The full dataset
      is returned in train, validation, test order. Releases without Parquet files quietly fall back to JSONL.
    - ``fmt="jsonl"``: downloads the JSONL file and returns its records verbatim (2025.1 and 2025.2 only).

    For columnar access, which is much faster and lighter on memory, use the Parquet files directly, e.g.,
    ``matpes.parquet.read_table(download(...))`` or the helpers in :mod:`matpes.convert`.

    Args:
        functional: "PBE" or "R2SCAN". Defaults to "PBE".
        version: Release. Defaults to :data:`LATEST_VERSION`.
        return_data: Whether to load and return the data. If False, the files are only downloaded.
        download_atoms: Whether to also download and return the isolated-atom reference energies.
        fmt: "parquet" (default) or "jsonl".
        split: Optional "train", "valid"/"validation" or "test" split instead of the full dataset.

    Returns:
        The dataset, or a tuple of (dataset, atomic reference energies) if ``download_atoms`` is True. None if
        ``return_data`` is False.
    """
    if fmt not in ("jsonl", "parquet"):
        raise ValueError(f"Unknown format {fmt!r}. Use 'parquet' or 'jsonl'.")
    data_path = download(functional, version, split=split, fmt=fmt)
    atoms_path = download(functional, version, atoms=True, fmt=fmt) if download_atoms else None

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
