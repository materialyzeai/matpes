"""Parquet format for MatPES.

MatPES records are stored as flat, Arrow-native columns rather than as serialized ``Structure.as_dict()`` blobs, so that
readers get columnar projection, predicate pushdown on row-group statistics, zero-copy access to numeric arrays and
good compression. One row is one structure. The columns (units in brackets) are:

========================== ======================================= ===================================================
Column                     Arrow type                              Description
========================== ======================================= ===================================================
matpes_id                  string                                  Unique MatPES identifier (null for isolated atoms).
functional                 string                                  "PBE" or "r2SCAN".
mp_id                      string                                  Source Materials Project ID.
formula_pretty             string                                  Reduced formula.
chemsys                    string                                  Alphabetically sorted, hyphen-separated elements.
nsites                     int32                                   Number of sites.
lattice                    fixed_size_list<float64, 9>             Row-major 3x3 lattice matrix [Å].
pbc                        fixed_size_list<bool, 3>                Periodic boundary conditions.
species                    list<string>                            Element symbol of each site.
positions                  list<fixed_size_list<float64, 3>>       Cartesian coordinates [Å].
forces                     list<fixed_size_list<float64, 3>>       DFT forces [eV/Å].
magmoms                    list<float64>                           Site magnetic moments from VASP [μB].
energy                     float64                                 DFT total energy [eV].
stress                     fixed_size_list<float64, 6>             DFT stress, Voigt (xx, yy, zz, yz, xz, xy), raw
                                                                   VASP convention [kbar, compressive positive].
bandgap                    float64                                 DFT band gap [eV].
cohesive_energy_per_atom   float64                                 [eV/atom].
formation_energy_per_atom  float64                                 [eV/atom]; null unless a relaxed ground state.
bader_charges              list<float64>                           Bader electron counts [e].
bader_magmoms              list<float64>                           Bader magnetic moments [μB].
cm5_partial_charges        list<float64>                           CM5 partial charges [e].
ddec6_partial_charges      list<float64>                           DDEC6 net atomic charges [e].
ddec6_spin_moments         list<float64>                           DDEC6 atomic spin moments [μB].
ddec6_bond_order_sums      list<float64>                           DDEC6 bond order sums.
ddec6_dipoles              list<fixed_size_list<float64, 3>>       DDEC6 atomic dipoles.
ddec6_rsquared_moments     list<float64>                           DDEC6 <r^2> moments [a.u.].
ddec6_rcubed_moments       list<float64>                           DDEC6 <r^3> moments [a.u.].
ddec6_rfourth_moments      list<float64>                           DDEC6 <r^4> moments [a.u.].
provenance                 struct                                  MD provenance (original_mp_id, MP version, MD
                                                                   ensemble/temperature/pressure/step, MLIP name).
========================== ======================================= ===================================================

Fields of the JSONL records that are cheaply derivable from the above (``composition``, ``symmetry``, ``volume``,
``density``, ``abs_forces``, ...) are dropped. Every function in this module that reads data accepts either a Parquet
file or a legacy MatPES ``.jsonl``/``.jsonl.gz`` file; JSONL records are converted on the fly with
:func:`record_to_row`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from monty.io import zopen

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from os import PathLike

    type Source = str | PathLike | pa.Table
    type Filters = pc.Expression | list | None

SCHEMA_VERSION = "1.0"

_VEC3 = pa.list_(pa.float64(), 3)
_PER_ATOM_FLOAT = pa.list_(pa.float64())
_PER_ATOM_VEC3 = pa.list_(_VEC3)

PROVENANCE_TYPE = pa.struct(
    [
        ("original_mp_id", pa.string()),
        ("materials_project_version", pa.string()),
        ("md_ensemble", pa.string()),
        ("md_temperature", pa.float64()),
        ("md_pressure", pa.float64()),
        ("md_step", pa.int64()),
        ("mlip_name", pa.string()),
    ]
)

DDEC6_FIELDS = (
    "partial_charges",
    "spin_moments",
    "bond_order_sums",
    "dipoles",
    "rsquared_moments",
    "rcubed_moments",
    "rfourth_moments",
)

SCHEMA = pa.schema(
    [
        ("matpes_id", pa.string()),
        ("functional", pa.string()),
        ("mp_id", pa.string()),
        ("formula_pretty", pa.string()),
        ("chemsys", pa.string()),
        ("nsites", pa.int32()),
        ("lattice", pa.list_(pa.float64(), 9)),
        ("pbc", pa.list_(pa.bool_(), 3)),
        ("species", pa.list_(pa.string())),
        ("positions", _PER_ATOM_VEC3),
        ("forces", _PER_ATOM_VEC3),
        ("magmoms", _PER_ATOM_FLOAT),
        ("energy", pa.float64()),
        ("stress", pa.list_(pa.float64(), 6)),
        ("bandgap", pa.float64()),
        ("cohesive_energy_per_atom", pa.float64()),
        ("formation_energy_per_atom", pa.float64()),
        ("bader_charges", _PER_ATOM_FLOAT),
        ("bader_magmoms", _PER_ATOM_FLOAT),
        ("cm5_partial_charges", _PER_ATOM_FLOAT),
        *[(f"ddec6_{k}", _PER_ATOM_VEC3 if k == "dipoles" else _PER_ATOM_FLOAT) for k in DDEC6_FIELDS],
        ("provenance", PROVENANCE_TYPE),
    ]
)
"""Arrow schema of a MatPES Parquet file."""

UNITS = {
    "lattice": "angstrom",
    "positions": "angstrom",
    "forces": "eV/angstrom",
    "magmoms": "bohr_magneton",
    "energy": "eV",
    "stress": "kbar (VASP sign convention, compressive positive; Voigt xx, yy, zz, yz, xz, xy)",
    "bandgap": "eV",
    "cohesive_energy_per_atom": "eV/atom",
    "formation_energy_per_atom": "eV/atom",
}

METADATA_KEY = b"matpes"
"""Key of the MatPES JSON entry in the Parquet file-level key-value metadata."""


_PARQUET_SUFFIXES = {".parquet", ".pq"}
_SPLIT_ORDER = ("train", "validation", "valid", "test")


def is_parquet(path: str | PathLike) -> bool:
    """Whether ``path`` is a Parquet file (by suffix) or a directory of Parquet files."""
    path = Path(path)
    if path.is_dir():
        return any(p.suffix.lower() in _PARQUET_SUFFIXES for p in path.iterdir())
    return path.suffix.lower() in _PARQUET_SUFFIXES


def parquet_files(path: str | PathLike) -> list[str]:
    """Return the Parquet files of a Parquet file or directory.

    Files in a directory are ordered train, validation, test, then any others by name, so that a directory of split
    files (as downloaded by :func:`matpes.data.download`) reads as one dataset in a deterministic order.

    Args:
        path: Parquet file or directory.

    Returns:
        List of file paths.
    """
    path = Path(path)
    if not path.is_dir():
        return [str(path)]

    def key(p: Path) -> tuple[int, str]:
        stem = p.stem.lower()
        return (_SPLIT_ORDER.index(stem) if stem in _SPLIT_ORDER else len(_SPLIT_ORDER), p.name)

    return [str(p) for p in sorted((p for p in path.iterdir() if p.suffix.lower() in _PARQUET_SUFFIXES), key=key)]


def record_to_row(record: dict) -> dict[str, Any]:
    """Convert a legacy MatPES JSON record into a flat row conforming to :data:`SCHEMA`.

    Args:
        record: One MatPES record as found in the ``.jsonl`` files, i.e., with a serialized pymatgen ``structure``.

    Returns:
        Dict keyed by :data:`SCHEMA` column names, with plain Python values.
    """
    struct = record["structure"]
    lattice = struct["lattice"]
    sites = struct["sites"]
    magmoms = [site.get("properties", {}).get("magmom") for site in sites]
    ddec6 = record.get("ddec6") or {}
    provenance = record.get("provenance")
    species = [site["species"][0]["element"] for site in sites]
    return {
        "matpes_id": record.get("matpes_id"),
        "functional": record.get("functional"),
        "mp_id": (provenance or {}).get("original_mp_id"),
        "formula_pretty": record.get("formula_pretty"),
        "chemsys": record.get("chemsys"),
        "nsites": len(sites),
        "lattice": [x for row in lattice["matrix"] for x in row],
        "pbc": list(lattice.get("pbc", (True, True, True))),
        "species": species,
        "positions": [site["xyz"] for site in sites],
        "forces": record["forces"],
        "magmoms": None if any(m is None for m in magmoms) else magmoms,
        "energy": record["energy"],
        "stress": record.get("stress"),
        "bandgap": record.get("bandgap"),
        "cohesive_energy_per_atom": record.get("cohesive_energy_per_atom"),
        "formation_energy_per_atom": record.get("formation_energy_per_atom"),
        "bader_charges": record.get("bader_charges"),
        "bader_magmoms": record.get("bader_magmoms"),
        "cm5_partial_charges": record.get("cm5_partial_charges"),
        **{f"ddec6_{k}": ddec6.get(k) for k in DDEC6_FIELDS},
        "provenance": provenance,
    }


def row_to_record(row: dict) -> dict[str, Any]:
    """Convert a flat row back into a legacy-style MatPES record with a serialized ``structure``.

    The stored fields are restored, plus the cheaply derived ``elements``, ``nelements``, ``composition``,
    ``composition_reduced``, ``formula_anonymous``, ``volume``, ``density_atomic`` and ``abs_forces``. ``symmetry``,
    ``density`` and ``builder_meta`` are not regenerated.

    Args:
        row: A row as yielded by :func:`iter_rows` (or :func:`record_to_row`).

    Returns:
        Dict in the layout of a MatPES JSONL record.
    """
    from .convert import row_to_structure  # noqa: PLC0415 (circular)

    def _tolist(value):
        return value.tolist() if isinstance(value, np.ndarray) else value

    record = {
        k: _tolist(row.get(k))
        for k in (
            "matpes_id",
            "functional",
            "formula_pretty",
            "chemsys",
            "nsites",
            "energy",
            "forces",
            "stress",
            "bandgap",
            "cohesive_energy_per_atom",
            "formation_energy_per_atom",
            "bader_charges",
            "bader_magmoms",
            "cm5_partial_charges",
            "provenance",
        )
    }
    structure = row_to_structure(row)
    record["structure"] = structure.as_dict()
    composition = structure.composition
    record["elements"] = sorted(str(el) for el in composition.elements)
    record["nelements"] = len(record["elements"])
    record["composition"] = composition.as_dict()
    record["composition_reduced"] = composition.reduced_composition.as_dict()
    record["formula_anonymous"] = composition.anonymized_formula
    record["volume"] = structure.volume
    record["density_atomic"] = structure.volume / len(structure)
    if row.get("forces") is not None:
        record["abs_forces"] = np.linalg.norm(np.asarray(row["forces"]), axis=1).tolist()
    ddec6 = {k: _tolist(row.get(f"ddec6_{k}")) for k in DDEC6_FIELDS}
    record["ddec6"] = None if all(v is None for v in ddec6.values()) else ddec6
    return record


def _iter_jsonl(path: str | PathLike) -> Iterator[dict]:
    with zopen(Path(path), "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _batched(it: Iterable, n: int) -> Iterator[list]:
    batch: list = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def _to_expression(filters: Filters) -> pc.Expression | None:
    if filters is None or isinstance(filters, pc.Expression):
        return filters
    return pq.filters_to_expression(filters)


def iter_batches(
    source: Source,
    columns: Sequence[str] | None = None,
    filters: Filters = None,
    batch_size: int = 4096,
) -> Iterator[pa.RecordBatch]:
    """Iterate over a MatPES dataset as Arrow record batches, in file order.

    Args:
        source: Path to a Parquet file, a directory of Parquet files (read in :func:`parquet_files` order), an
            (optionally gzipped) JSONL file, or an in-memory :class:`pyarrow.Table`.
        columns: Columns to read. Defaults to all. For Parquet, unread columns are never decoded.
        filters: Row filter, either a :class:`pyarrow.compute.Expression` (e.g., ``pc.field("chemsys") == "Fe-O"``)
            or DNF tuples as accepted by :func:`pyarrow.parquet.read_table` (e.g., ``[("nsites", "<", 10)]``).
            For Parquet this prunes whole row groups using column statistics.
        batch_size: Maximum number of rows per batch.

    Yields:
        :class:`pyarrow.RecordBatch` objects conforming to (a projection of) :data:`SCHEMA`.
    """
    expr = _to_expression(filters)
    columns = list(columns) if columns is not None else None
    if isinstance(source, pa.Table):
        table = source.filter(expr) if expr is not None else source
        if columns is not None:
            table = table.select(columns)
        yield from table.to_batches(max_chunksize=batch_size)
        return
    if is_parquet(source):
        dataset = pds.dataset(parquet_files(source), format="parquet")
        yield from dataset.to_batches(columns=columns, filter=expr, batch_size=batch_size)
        return
    for records in _batched(_iter_jsonl(source), batch_size):
        table = pa.Table.from_pylist([record_to_row(r) for r in records], schema=SCHEMA)
        if expr is not None:
            table = table.filter(expr)
        if columns is not None:
            table = table.select(columns)
        yield from table.to_batches()


def read_table(source: Source, columns: Sequence[str] | None = None, filters: Filters = None) -> pa.Table:
    """Read a MatPES dataset into a :class:`pyarrow.Table`.

    Use ``table.to_pandas()`` or ``polars.from_arrow(table)`` for dataframe access.

    Args:
        source: Path to a Parquet file or directory, a JSONL file, or a :class:`pyarrow.Table`.
        columns: Columns to read. Defaults to all.
        filters: Row filter. See :func:`iter_batches`.

    Returns:
        :class:`pyarrow.Table`.
    """
    if not isinstance(source, pa.Table) and is_parquet(source):
        files = parquet_files(source)
        return pq.read_table(files if len(files) > 1 else files[0], columns=columns, filters=_to_expression(filters))
    batches = list(iter_batches(source, columns=columns, filters=filters))
    if batches:
        return pa.Table.from_batches(batches)
    schema = SCHEMA if columns is None else pa.schema([SCHEMA.field(c) for c in columns])
    return schema.empty_table()


def _split_list(arr: pa.Array, width: int | None) -> list[np.ndarray | None]:
    """Split a (possibly sliced) list<float64> or list<fixed_size_list<float64, width>> array into numpy views."""
    offsets = arr.offsets.to_numpy()
    # Slice the child array directly rather than using flatten(), which skips the (possibly non-empty) value ranges
    # of null entries and would misalign the offsets.
    values = arr.values.slice(offsets[0], offsets[-1] - offsets[0])
    offsets = offsets - offsets[0]
    if width is not None:
        values = values.flatten()
    flat = values.to_numpy(zero_copy_only=False)
    if width is not None:
        flat = flat.reshape(-1, width)
    valid = arr.is_valid().to_numpy(zero_copy_only=False)
    return [flat[offsets[i] : offsets[i + 1]] if valid[i] else None for i in range(len(arr))]


def batch_to_rows(batch: pa.RecordBatch | pa.Table) -> list[dict[str, Any]]:
    """Convert an Arrow batch into a list of row dicts.

    Numeric per-atom and fixed-size columns are returned as numpy arrays (views into the Arrow buffers where
    possible): ``lattice`` (3, 3), ``positions``/``forces``/``ddec6_dipoles`` (nsites, 3), ``stress`` (6,), per-atom
    scalars (nsites,). Other columns are plain Python values.

    Args:
        batch: Record batch or table conforming to (a projection of) :data:`SCHEMA`.

    Returns:
        List of dicts, one per row.
    """
    if isinstance(batch, pa.Table):
        batch = batch.combine_chunks().to_batches()[0] if batch.num_rows else pa.RecordBatch.from_pylist([])
    n = batch.num_rows
    cols: dict[str, list] = {}
    for name in batch.schema.names:
        arr = batch.column(name)
        typ = arr.type
        if pa.types.is_fixed_size_list(typ) and pa.types.is_floating(typ.value_type):
            size = typ.list_size
            flat = arr.flatten().to_numpy(zero_copy_only=False).reshape(-1, size)
            valid = arr.is_valid().to_numpy(zero_copy_only=False)
            if name == "lattice":
                flat = flat.reshape(-1, 3, 3)
            cols[name] = [flat[i] if valid[i] else None for i in range(n)]
        elif pa.types.is_list(typ) and pa.types.is_floating(typ.value_type):
            cols[name] = _split_list(arr, None)
        elif (
            pa.types.is_list(typ)
            and pa.types.is_fixed_size_list(typ.value_type)
            and pa.types.is_floating(typ.value_type.value_type)
        ):
            cols[name] = _split_list(arr, typ.value_type.list_size)
        else:
            cols[name] = arr.to_pylist()
    return [{k: v[i] for k, v in cols.items()} for i in range(n)]


def iter_rows(
    source: Source,
    columns: Sequence[str] | None = None,
    filters: Filters = None,
    batch_size: int = 4096,
) -> Iterator[dict[str, Any]]:
    """Iterate over a MatPES dataset row by row with constant memory.

    Args:
        source: Path to a Parquet or JSONL file, or a :class:`pyarrow.Table`.
        columns: Columns to read. Defaults to all.
        filters: Row filter. See :func:`iter_batches`.
        batch_size: Number of rows decoded at a time.

    Yields:
        Row dicts as described in :func:`batch_to_rows`.
    """
    for batch in iter_batches(source, columns=columns, filters=filters, batch_size=batch_size):
        yield from batch_to_rows(batch)


def read_metadata(path: str | PathLike) -> dict[str, Any]:
    """Read the MatPES file-level metadata (dataset version, functional, units, ...) of a Parquet file or directory.

    For a directory, the metadata of its first file is returned.
    """
    kv = pq.read_schema(parquet_files(path)[0]).metadata or {}
    return json.loads(kv[METADATA_KEY]) if METADATA_KEY in kv else {}


def write_parquet(
    rows: Iterable[dict] | pa.Table,
    path: str | PathLike,
    metadata: dict[str, Any] | None = None,
    row_group_size: int = 10_000,
    compression: str = "zstd",
) -> int:
    """Write rows conforming to :data:`SCHEMA` to a Parquet file, streaming in row groups.

    Args:
        rows: Iterable of row dicts (e.g., from :func:`record_to_row`) or a :class:`pyarrow.Table`.
        path: Output path.
        metadata: Extra entries for the MatPES file-level metadata (e.g., ``{"version": "2025.3"}``).
        row_group_size: Rows per row group. ~10k rows keeps row groups well below the Hugging Face viewer limits
            while retaining effective statistics-based pruning.
        compression: Parquet compression codec.

    Pages are cut with content-defined chunking, so that a new release that leaves most rows unchanged shares most of
    its bytes with the previous one and Hugging Face's deduplicating storage uploads only the changed parts.

    Returns:
        Number of rows written.
    """
    meta = {"schema_version": SCHEMA_VERSION, "units": UNITS, **(metadata or {})}
    schema = SCHEMA.with_metadata({METADATA_KEY: json.dumps(meta).encode()})
    n = 0
    with pq.ParquetWriter(
        path, schema, compression=compression, write_page_index=True, use_content_defined_chunking=True
    ) as writer:
        if isinstance(rows, pa.Table):
            writer.write_table(rows.select(SCHEMA.names).cast(SCHEMA), row_group_size=row_group_size)
            return rows.num_rows
        for chunk in _batched(rows, row_group_size):
            writer.write_table(pa.Table.from_pylist(chunk, schema=schema), row_group_size=row_group_size)
            n += len(chunk)
    return n


def jsonl_to_parquet(
    src: str | PathLike,
    dst: str | PathLike | None = None,
    metadata: dict[str, Any] | None = None,
    row_group_size: int = 10_000,
    compression: str = "zstd",
) -> Path:
    """Convert a MatPES JSONL file into Parquet with constant memory.

    Row order is preserved.

    Args:
        src: Input ``.jsonl`` or ``.jsonl.gz`` file.
        dst: Output path. Defaults to ``src`` with the ``.jsonl[.gz]`` suffix replaced by ``.parquet``.
        metadata: Extra file-level metadata. The source file name is always recorded.
        row_group_size: Rows per row group.
        compression: Parquet compression codec.

    Returns:
        Path of the written Parquet file.
    """
    src = Path(src)
    if dst is None:
        stem = src.name.removesuffix(".gz").removesuffix(".jsonl").removesuffix(".json")
        dst = src.with_name(f"{stem}.parquet")
    dst = Path(dst)
    meta = {"source": src.name, **(metadata or {})}
    write_parquet(
        (record_to_row(r) for r in _iter_jsonl(src)),
        dst,
        metadata=meta,
        row_group_size=row_group_size,
        compression=compression,
    )
    return dst
