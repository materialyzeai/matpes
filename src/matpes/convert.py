"""Convert MatPES data into pymatgen Structures, ASE Atoms, MatGL datasets and LMDB databases.

All functions accept a MatPES Parquet file, a legacy ``.jsonl``/``.jsonl.gz`` file or an in-memory
:class:`pyarrow.Table`, and optional column ``filters`` (see :func:`matpes.parquet.iter_batches`). For example::

    import pyarrow.compute as pc
    from matpes.convert import to_structures, to_mgl_dataset

    structures = to_structures("MatPES-R2SCAN-2025.3.parquet", filters=pc.field("chemsys") == "Li-O")
    dataset = to_mgl_dataset("MatPES-R2SCAN-2025.3-train.parquet", include_stress=True)

Optional dependencies: ``ase`` for :func:`row_to_atoms`, ``matgl`` for the MatGL helpers and ``lmdb`` for the LMDB
helpers (``pip install "matpes[ase,matgl,lmdb]"``).
"""

from __future__ import annotations

import functools
import json
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pymatgen.core import Lattice, Structure

from .parquet import iter_rows

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from os import PathLike

    from ase import Atoms

    from .parquet import Filters, Source

EV_PER_ANG3_PER_GPA = 1 / 160.21766208
"""1 GPa in eV/Å^3."""

KBAR_TO_MATGL_GPA = -0.1
"""Factor converting raw VASP stress (kbar, compressive positive) to MatGL's GPa, compressive negative."""

STRUCTURE_COLUMNS = ("lattice", "pbc", "species", "positions", "magmoms")
PES_COLUMNS = (*STRUCTURE_COLUMNS, "energy", "forces", "stress")
INFO_COLUMNS = (
    "matpes_id",
    "functional",
    "mp_id",
    "formula_pretty",
    "bandgap",
    "cohesive_energy_per_atom",
    "formation_energy_per_atom",
)

ChargesSource = Literal["ddec6", "bader", "cm5"]
MagmomsSource = Literal["ddec6", "bader", "vasp"]

_CHARGE_COLUMNS = {"ddec6": "ddec6_partial_charges", "bader": "bader_charges", "cm5": "cm5_partial_charges"}
_MAGMOM_COLUMNS = {"ddec6": "ddec6_spin_moments", "bader": "bader_magmoms", "vasp": "magmoms"}


def row_to_structure(row: dict) -> Structure:
    """Build a pymatgen Structure from a MatPES row.

    Site magnetic moments, when present, are stored as the ``magmom`` site property.

    Args:
        row: Row dict from :func:`matpes.parquet.iter_rows` (needs ``lattice``, ``species`` and ``positions``).

    Returns:
        Structure.
    """
    lattice = np.asarray(row["lattice"], dtype=float).reshape(3, 3)
    pbc = tuple(bool(x) for x in row.get("pbc") or (True, True, True))
    magmoms = row.get("magmoms")
    return Structure(
        Lattice(lattice, pbc=pbc),  # type: ignore[arg-type]
        row["species"],
        np.asarray(row["positions"], dtype=float),
        coords_are_cartesian=True,
        site_properties=None if magmoms is None else {"magmom": list(magmoms)},
    )


def row_to_atoms(row: dict) -> Atoms:
    """Build an ASE Atoms object from a MatPES row.

    If the row has PES labels, they are attached through a :class:`~ase.calculators.singlepoint.SinglePointCalculator`
    in ASE units and sign convention: energy (eV), forces (eV/Å), stress (eV/Å^3, Voigt, tensile positive, i.e. the raw
    VASP kbar value times -0.1 GPa) and magmoms (μB). Scalar metadata (``matpes_id``, ``functional``, ``mp_id``, ...)
    present in the row go into ``atoms.info``.

    Args:
        row: Row dict from :func:`matpes.parquet.iter_rows`.

    Returns:
        ase.Atoms.
    """
    from ase import Atoms  # noqa: PLC0415
    from ase.calculators.singlepoint import SinglePointCalculator  # noqa: PLC0415

    atoms = Atoms(
        symbols=row["species"],
        positions=np.asarray(row["positions"], dtype=float),
        cell=np.asarray(row["lattice"], dtype=float).reshape(3, 3),
        pbc=[bool(x) for x in row.get("pbc") or (True, True, True)],
    )
    atoms.info.update({k: row[k] for k in INFO_COLUMNS if row.get(k) is not None})
    results: dict[str, Any] = {}
    if row.get("energy") is not None:
        results["energy"] = float(row["energy"])
    if row.get("forces") is not None:
        results["forces"] = np.asarray(row["forces"], dtype=float)
    if row.get("stress") is not None:
        results["stress"] = np.asarray(row["stress"], dtype=float) * KBAR_TO_MATGL_GPA * EV_PER_ANG3_PER_GPA
    if row.get("magmoms") is not None:
        results["magmoms"] = np.asarray(row["magmoms"], dtype=float)
        results["magmom"] = float(np.sum(results["magmoms"]))
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def _columns(base: Sequence[str], extra: Sequence[str] = ()) -> list[str]:
    return list(dict.fromkeys([*base, *extra]))


def iter_structures(source: Source, filters: Filters = None) -> Iterator[Structure]:
    """Lazily yield pymatgen Structures (with ``magmom`` site properties) from a MatPES dataset."""
    for row in iter_rows(source, columns=STRUCTURE_COLUMNS, filters=filters):
        yield row_to_structure(row)


def to_structures(source: Source, filters: Filters = None) -> list[Structure]:
    """Load pymatgen Structures from a MatPES dataset.

    Args:
        source: Parquet or JSONL path, or a :class:`pyarrow.Table`.
        filters: Row filter. See :func:`matpes.parquet.iter_batches`.

    Returns:
        List of Structures, in file order.
    """
    return list(iter_structures(source, filters=filters))


def iter_atoms(source: Source, filters: Filters = None) -> Iterator[Atoms]:
    """Lazily yield ASE Atoms with attached PES labels from a MatPES dataset. See :func:`row_to_atoms`."""
    for row in iter_rows(source, columns=_columns(PES_COLUMNS, INFO_COLUMNS), filters=filters):
        yield row_to_atoms(row)


def to_atoms(source: Source, filters: Filters = None) -> list[Atoms]:
    """Load ASE Atoms with attached PES labels from a MatPES dataset. See :func:`row_to_atoms`.

    Args:
        source: Parquet or JSONL path, or a :class:`pyarrow.Table`.
        filters: Row filter. See :func:`matpes.parquet.iter_batches`.

    Returns:
        List of Atoms, in file order.
    """
    return list(iter_atoms(source, filters=filters))


def _iter_mgl_samples(
    source: Source,
    filters: Filters,
    include_stress: bool,
    include_charges: bool | ChargesSource,
    include_magmoms: bool | MagmomsSource,
) -> Iterator[tuple[Structure, dict[str, Any]]]:
    """Yield (structure, labels) pairs in MatGL conventions, skipping rows lacking a requested optional label."""
    optional: dict[str, str] = {}
    if include_charges:
        optional["charges"] = _CHARGE_COLUMNS["ddec6" if include_charges is True else include_charges]
    if include_magmoms:
        optional["magmoms"] = _MAGMOM_COLUMNS["ddec6" if include_magmoms is True else include_magmoms]
    columns = _columns(PES_COLUMNS, list(optional.values()))
    for row in iter_rows(source, columns=columns, filters=filters):
        if any(row[col] is None for col in optional.values()):
            continue
        labels: dict[str, Any] = {"energies": float(row["energy"]), "forces": np.asarray(row["forces"]).tolist()}
        if include_stress:
            labels["stresses"] = (np.asarray(row["stress"]) * KBAR_TO_MATGL_GPA).tolist()
        for key, col in optional.items():
            labels[key] = np.asarray(row[col]).tolist()
        yield row_to_structure(row), labels


def _default_converter(converter, cutoff: float, element_types: Sequence[str] | None):
    if converter is not None:
        return converter
    from matgl.config import DEFAULT_ELEMENTS  # noqa: PLC0415
    from matgl.ext.pymatgen import Structure2Graph  # noqa: PLC0415

    return Structure2Graph(element_types=tuple(element_types or DEFAULT_ELEMENTS), cutoff=cutoff)


def to_mgl_dataset(
    source: Source,
    filters: Filters = None,
    converter=None,
    cutoff: float = 5.0,
    element_types: Sequence[str] | None = None,
    include_stress: bool = True,
    include_charges: bool | ChargesSource = False,
    include_magmoms: bool | MagmomsSource = False,
    **kwargs,
):
    """Build an in-memory :class:`matgl.graph.data.MGLDataset` for PES training from a MatPES dataset.

    Labels follow MatGL conventions: ``energies`` (eV), ``forces`` (eV/Å), ``stresses`` (Voigt, GPa, compressive
    negative, i.e. raw VASP kbar times -0.1), and optionally per-atom ``charges`` and ``magmoms``. Rows where a
    requested optional label is null (failed charge analysis) are skipped. For the full dataset, prefer
    :func:`write_mgl_shards`, which streams to disk with constant memory.

    Args:
        source: Parquet or JSONL path, or a :class:`pyarrow.Table`.
        filters: Row filter. See :func:`matpes.parquet.iter_batches`.
        converter: MatGL graph converter. Defaults to ``Structure2Graph(DEFAULT_ELEMENTS, cutoff)``. When fine-tuning,
            the element list must match ``model.element_types``.
        cutoff: Graph cutoff radius (Å) for the default converter.
        element_types: Element list for the default converter. Defaults to ``matgl.config.DEFAULT_ELEMENTS``.
        include_stress: Whether to include ``stresses``.
        include_charges: ``True`` or ``"ddec6"`` (DDEC6 net charges), ``"bader"`` or ``"cm5"`` to add ``charges``.
        include_magmoms: ``True`` or ``"ddec6"`` (DDEC6 spin moments), ``"bader"`` or ``"vasp"`` to add ``magmoms``.
        **kwargs: Passed to :class:`~matgl.graph.data.MGLDataset` (e.g., ``root``, ``include_line_graph``).

    Returns:
        matgl.graph.data.MGLDataset.
    """
    from matgl.graph.data import MGLDataset  # noqa: PLC0415

    structures: list[Structure] = []
    labels: dict[str, list] = {}
    for structure, sample in _iter_mgl_samples(source, filters, include_stress, include_charges, include_magmoms):
        structures.append(structure)
        for key, value in sample.items():
            labels.setdefault(key, []).append(value)
    return MGLDataset(
        structures=structures,
        labels=labels,
        converter=_default_converter(converter, cutoff, element_types),
        **kwargs,
    )


def write_mgl_shards(
    source: Source,
    root: str | PathLike,
    filters: Filters = None,
    converter=None,
    cutoff: float = 5.0,
    element_types: Sequence[str] | None = None,
    include_stress: bool = True,
    include_charges: bool | ChargesSource = False,
    include_magmoms: bool | MagmomsSource = False,
    shard_size: int = 1000,
    **kwargs,
) -> Path:
    """Stream a MatPES dataset into on-disk MatGL graph shards, readable with :class:`matgl.graph.data.MGLDiskDataset`.

    Memory use is bounded by ``shard_size`` regardless of dataset size. Labels are as in :func:`to_mgl_dataset`.

    Args:
        source: Parquet or JSONL path, or a :class:`pyarrow.Table`.
        root: Output directory.
        filters: Row filter. See :func:`matpes.parquet.iter_batches`.
        converter: MatGL graph converter. See :func:`to_mgl_dataset`.
        cutoff: Graph cutoff radius (Å) for the default converter.
        element_types: Element list for the default converter.
        include_stress: Whether to include ``stresses``.
        include_charges: See :func:`to_mgl_dataset`.
        include_magmoms: See :func:`to_mgl_dataset`.
        shard_size: Structures per shard.
        **kwargs: Passed to :func:`matgl.graph.data.write_mgl_shards`.

    Returns:
        Path to ``root``.
    """
    from matgl.graph.data import write_mgl_shards as _write  # noqa: PLC0415

    records = (
        {"structure": structure, "labels": labels}
        for structure, labels in _iter_mgl_samples(source, filters, include_stress, include_charges, include_magmoms)
    )
    _write(
        records,
        root,
        converter=_default_converter(converter, cutoff, element_types),
        shard_size=shard_size,
        **kwargs,
    )
    return Path(root)


_LMDB_LEN_KEY = b"__len__"
_LMDB_META_KEY = b"__metadata__"


def to_lmdb(
    source: Source,
    path: str | PathLike,
    filters: Filters = None,
    columns: Sequence[str] | None = None,
    map_size: int = 2**40,
    commit_every: int = 10_000,
) -> int:
    """Write a MatPES dataset to a single-file LMDB database for fast random access during training.

    Layout: key ``str(i).encode()`` for ``i in range(n)`` maps to a pickled row dict (numpy arrays for numeric per-atom
    columns, as from :func:`matpes.parquet.iter_rows`); ``b"__len__"`` holds ``n`` and ``b"__metadata__"`` a JSON
    description. Read it back with :class:`MatPESLMDB`. As with any pickle-based store, only open LMDB files from
    trusted sources.

    Args:
        source: Parquet or JSONL path, or a :class:`pyarrow.Table`.
        path: Output LMDB file. Must not exist.
        filters: Row filter. See :func:`matpes.parquet.iter_batches`.
        columns: Columns to store. Defaults to all.
        map_size: Maximum database size in bytes. Sparse on Linux/macOS, so the 1 TiB default costs nothing.
        commit_every: Rows per write transaction.

    Returns:
        Number of rows written.
    """
    import lmdb  # noqa: PLC0415

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists.")
    env = lmdb.open(str(path), map_size=map_size, subdir=False, meminit=False, map_async=True, lock=True)
    n = 0
    try:
        txn = env.begin(write=True)
        for row in iter_rows(source, columns=columns, filters=filters):
            txn.put(str(n).encode(), pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL))
            n += 1
            if n % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
        meta = {"format": "matpes-lmdb", "version": 1, "columns": list(columns) if columns else None}
        txn.put(_LMDB_LEN_KEY, str(n).encode())
        txn.put(_LMDB_META_KEY, json.dumps(meta).encode())
        txn.commit()
        env.sync()
    finally:
        env.close()
    return n


@functools.cache
def _open_lmdb_readonly(path: str, pid: int):
    """Open (once per process) a read-only LMDB environment. lmdb forbids opening the same file twice in a process."""
    import lmdb  # noqa: PLC0415

    return lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)


class MatPESLMDB:
    """Random-access reader for an LMDB written by :func:`to_lmdb`.

    Behaves as a sequence of row dicts and is safe to use as a map-style PyTorch dataset with multiple ``DataLoader``
    workers: the environment is opened lazily in each process.
    """

    def __init__(self, path: str | PathLike):
        """
        Args:
            path: LMDB file written by :func:`to_lmdb`.
        """
        self.path = Path(path)
        self._env = None
        self._pid: int | None = None
        with self._txn() as txn:
            self._len = int(txn.get(_LMDB_LEN_KEY))
            self.metadata = json.loads(txn.get(_LMDB_META_KEY))

    def _txn(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            self._env = _open_lmdb_readonly(str(self.path.resolve()), pid)
            self._pid = pid
        return self._env.begin(write=False)  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self._len
        if not 0 <= idx < self._len:
            raise IndexError(idx)
        with self._txn() as txn:
            return pickle.loads(txn.get(str(idx).encode()))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(self._len):
            yield self[i]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_pid"] = None
        return state

    def get_structure(self, idx: int) -> Structure:
        """Return entry ``idx`` as a pymatgen Structure."""
        return row_to_structure(self[idx])

    def get_atoms(self, idx: int) -> Atoms:
        """Return entry ``idx`` as an ASE Atoms object with attached PES labels."""
        return row_to_atoms(self[idx])
