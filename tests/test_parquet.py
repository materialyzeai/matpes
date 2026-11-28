from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pymatgen.core import Structure

from matpes.convert import (
    EV_PER_ANG3_PER_GPA,
    MatPESLMDB,
    row_to_atoms,
    row_to_structure,
    to_atoms,
    to_lmdb,
    to_structures,
)
from matpes.data import get_repo_files
from matpes.parquet import (
    SCHEMA,
    batch_to_rows,
    is_parquet,
    iter_rows,
    jsonl_to_parquet,
    parquet_files,
    read_metadata,
    read_table,
    record_to_row,
    row_to_record,
)

SAMPLE = Path(__file__).parent / "files" / "MatPES-PBE-sample.jsonl"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    with SAMPLE.open() as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def parquet_file(tmp_path_factory) -> Path:
    return jsonl_to_parquet(SAMPLE, tmp_path_factory.mktemp("pq") / "sample.parquet", metadata={"version": "test"})


@pytest.fixture(params=["jsonl", "parquet", "table"])
def source(request, parquet_file):
    """Every reader must accept JSONL, Parquet and in-memory tables interchangeably."""
    return {"jsonl": SAMPLE, "parquet": parquet_file, "table": pq.read_table(parquet_file)}[request.param]


def test_schema_is_flat_and_arrow_native(parquet_file):
    schema = pq.read_schema(parquet_file)
    assert schema.remove_metadata() == SCHEMA
    assert schema.field("lattice").type == pa.list_(pa.float64(), 9)
    assert schema.field("stress").type == pa.list_(pa.float64(), 6)
    assert schema.field("positions").type == pa.list_(pa.list_(pa.float64(), 3))
    assert "structure" not in schema.names
    meta = read_metadata(parquet_file)
    assert meta["version"] == "test"
    assert meta["source"] == SAMPLE.name
    assert meta["units"]["stress"].startswith("kbar")


def test_all_floats_are_float64(parquet_file):
    def leaves(typ):
        if pa.types.is_struct(typ):
            for i in range(typ.num_fields):
                yield from leaves(typ.field(i).type)
        elif pa.types.is_list(typ) or pa.types.is_fixed_size_list(typ):
            yield from leaves(typ.value_type)
        else:
            yield typ

    floats = [t for field in pq.read_schema(parquet_file) for t in leaves(field.type) if pa.types.is_floating(t)]
    assert floats
    assert all(t == pa.float64() for t in floats)
    row = next(iter_rows(parquet_file))
    for key in ("lattice", "positions", "forces", "stress", "magmoms", "ddec6_dipoles"):
        assert row[key].dtype == np.float64, key


def test_roundtrip_values(records, parquet_file):
    table = pq.read_table(parquet_file)
    assert table.num_rows == len(records)
    rows = batch_to_rows(table)
    for rec, row in zip(records, rows, strict=True):
        ref = Structure.from_dict(rec["structure"])
        struct = row_to_structure(row)
        assert np.allclose(struct.lattice.matrix, ref.lattice.matrix, atol=0, rtol=0)
        assert np.allclose(struct.cart_coords, ref.cart_coords, atol=1e-12)
        assert np.allclose(struct.frac_coords, ref.frac_coords, atol=1e-10)
        assert [str(s) for s in struct.species] == [str(s) for s in ref.species]
        assert struct.site_properties["magmom"] == ref.site_properties["magmom"]
        np.testing.assert_array_equal(row["forces"], rec["forces"])
        np.testing.assert_array_equal(row["stress"], rec["stress"])
        assert row["energy"] == rec["energy"]
        assert row["matpes_id"] == rec["matpes_id"]
        assert row["mp_id"] == (rec["provenance"] or {}).get("original_mp_id")
        assert row["provenance"] == rec["provenance"]
        assert row["formation_energy_per_atom"] == rec["formation_energy_per_atom"]
        ddec6 = rec.get("ddec6")
        if ddec6 is None:
            assert row["ddec6_partial_charges"] is None
            assert row["ddec6_dipoles"] is None
        else:
            np.testing.assert_array_equal(row["ddec6_partial_charges"], ddec6["partial_charges"])
            np.testing.assert_array_equal(row["ddec6_dipoles"], ddec6["dipoles"])
        if rec.get("bader_charges") is None:
            assert row["bader_charges"] is None
        else:
            np.testing.assert_array_equal(row["bader_charges"], rec["bader_charges"])


def test_null_charges_present_in_sample(records):
    # The fixture must exercise null per-atom list columns.
    assert any(r.get("ddec6") is None for r in records)


def test_row_to_record(records, parquet_file):
    for rec, row in zip(records, iter_rows(parquet_file), strict=True):
        back = row_to_record(row)
        assert Structure.from_dict(back["structure"]) == Structure.from_dict(rec["structure"])
        assert back["forces"] == rec["forces"]
        assert back["ddec6"] == rec.get("ddec6")
        json.dumps(back)  # must be JSON serializable


def test_readers_accept_any_source(source, records):
    structures = to_structures(source)
    assert len(structures) == len(records)
    assert structures[0] == Structure.from_dict(records[0]["structure"])
    table = read_table(source, columns=["matpes_id", "energy"])
    assert table.column_names == ["matpes_id", "energy"]
    assert table.column("energy").to_pylist() == [r["energy"] for r in records]


@pytest.mark.parametrize(
    "filters",
    [
        pc.field("nsites") == 1,
        [("nsites", "=", 1)],
    ],
)
def test_filters(source, records, filters):
    expected = [r["formula_pretty"] for r in records if r["nsites"] == 1]
    rows = list(iter_rows(source, columns=["formula_pretty"], filters=filters))
    assert [r["formula_pretty"] for r in rows] == expected
    assert expected


def test_filter_no_match(source):
    table = read_table(source, filters=pc.field("chemsys") == "Xx-Yy")
    assert table.num_rows == 0
    assert table.schema.names == SCHEMA.names


def test_sliced_batches(parquet_file, records):
    # Small batches exercise non-zero Arrow offsets in list columns.
    rows = list(iter_rows(parquet_file, batch_size=3))
    assert [r["matpes_id"] for r in rows] == [r["matpes_id"] for r in records]
    for rec, row in zip(records, rows, strict=True):
        np.testing.assert_array_equal(row["positions"], [s["xyz"] for s in rec["structure"]["sites"]])


def test_atoms(source, records):
    atoms_list = to_atoms(source)
    assert len(atoms_list) == len(records)
    for rec, atoms in zip(records, atoms_list, strict=True):
        assert atoms.get_chemical_symbols() == [s["species"][0]["element"] for s in rec["structure"]["sites"]]
        assert atoms.get_potential_energy() == pytest.approx(rec["energy"])
        np.testing.assert_allclose(atoms.get_forces(), rec["forces"])
        # ASE: eV/A^3, tensile positive. VASP: kbar, compressive positive.
        np.testing.assert_allclose(atoms.get_stress(), -0.1 * np.array(rec["stress"]) * EV_PER_ANG3_PER_GPA)
        assert atoms.info.get("matpes_id") == rec["matpes_id"]
        assert atoms.info["functional"] == "PBE"


def test_row_to_atoms_without_labels(parquet_file):
    row = next(iter_rows(parquet_file, columns=["lattice", "pbc", "species", "positions"]))
    atoms = row_to_atoms(row)
    assert atoms.calc is None


def test_lmdb(tmp_path, parquet_file, records):
    path = tmp_path / "sample.lmdb"
    n = to_lmdb(parquet_file, path, commit_every=2)
    assert n == len(records)
    with pytest.raises(FileExistsError):
        to_lmdb(parquet_file, path)
    db = MatPESLMDB(path)
    assert len(db) == len(records)
    assert db[0]["matpes_id"] == records[0]["matpes_id"]
    assert db[-1]["energy"] == records[-1]["energy"]
    assert db.get_structure(1) == Structure.from_dict(records[1]["structure"])
    assert db.get_atoms(2).get_potential_energy() == pytest.approx(records[2]["energy"])
    assert len(list(db)) == len(records)
    with pytest.raises(IndexError):
        db[len(records)]
    clone = pickle.loads(pickle.dumps(db))
    assert clone[0]["matpes_id"] == records[0]["matpes_id"]


def test_lmdb_columns_and_filters(tmp_path):
    path = tmp_path / "subset.lmdb"
    n = to_lmdb(SAMPLE, path, columns=["matpes_id", "energy"], filters=pc.field("nsites") > 1)
    db = MatPESLMDB(path)
    assert len(db) == n > 0
    assert set(db[0]) == {"matpes_id", "energy"}
    assert db.metadata["columns"] == ["matpes_id", "energy"]


def test_record_to_row_matches_schema(records):
    table = pa.Table.from_pylist([record_to_row(r) for r in records], schema=SCHEMA)
    assert table.num_rows == len(records)


def test_get_repo_files():
    splits = ["pbe/train.parquet", "pbe/validation.parquet", "pbe/test.parquet"]
    assert get_repo_files("pbe") == ("v2025.3", splits)
    assert get_repo_files("r2scan", split="train") == ("v2025.3", ["r2scan/train.parquet"])
    assert get_repo_files("R2SCAN", split="valid") == ("v2025.3", ["r2scan/validation.parquet"])
    assert get_repo_files("PBE", atoms=True) == ("v2025.3", ["atoms/pbe.parquet"])
    # Releases without Parquet files quietly fall back to JSONL at their own revision; atoms stay Parquet.
    assert get_repo_files("PBE", version="2025.2") == ("v2025.2", ["MatPES-PBE-2025.2.jsonl"])
    assert get_repo_files("R2SCAN", version="2025.2", split="validation") == (
        "v2025.2",
        ["MatPES-R2SCAN-2025.2-valid.jsonl"],
    )
    assert get_repo_files("PBE", version="2025.1", atoms=True) == ("v2025.3", ["atoms/pbe.parquet"])
    # JSONL on request for the JSONL releases only.
    assert get_repo_files("PBE", version="2025.1", fmt="jsonl") == ("v2025.1", ["MatPES-PBE-2025.1.jsonl"])
    assert get_repo_files("PBE", version="2025.2", atoms=True, fmt="jsonl") == ("v2025.2", ["MatPES-PBE-atoms.jsonl"])
    with pytest.raises(ValueError, match="only available as Parquet"):
        get_repo_files("pbe", fmt="jsonl")
    with pytest.raises(ValueError, match="Unknown format"):
        get_repo_files("PBE", fmt="csv")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown split"):
        get_repo_files("PBE", split="dev")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown MatPES release"):
        get_repo_files("PBE", version="2024.1")


def test_parquet_directory(parquet_file, tmp_path):
    """A directory of split files reads as one dataset, ordered train, validation, test."""
    table = pq.read_table(parquet_file)
    parts = {"test": table.slice(0, 1), "train": table.slice(1, 3), "validation": table.slice(4)}
    for name, part in parts.items():
        pq.write_table(part.replace_schema_metadata(table.schema.metadata), tmp_path / f"{name}.parquet")
    assert is_parquet(tmp_path)
    assert [Path(f).stem for f in parquet_files(tmp_path)] == ["train", "validation", "test"]
    expected = pa.concat_tables([parts["train"], parts["validation"], parts["test"]])
    assert read_table(tmp_path).column("matpes_id").to_pylist() == expected.column("matpes_id").to_pylist()
    assert [r["matpes_id"] for r in iter_rows(tmp_path, columns=["matpes_id"])] == expected.column(
        "matpes_id"
    ).to_pylist()
    assert read_metadata(tmp_path)["version"] == "test"
    assert not is_parquet(tmp_path / "missing.jsonl")


@pytest.mark.parametrize("fmt", ["parquet", "jsonl"])
@pytest.mark.parametrize("stored", ["parquet", "jsonl"])
def test_get_data_returns_legacy_records(monkeypatch, parquet_file, records, fmt, stored):
    """get_data returns legacy dicts with d["structure"] whichever file format is downloaded."""
    import matpes.data  # noqa: PLC0415

    path = parquet_file if stored == "parquet" else SAMPLE
    requested = []

    def fake_download(*args, **kwargs):
        requested.append(kwargs.get("fmt"))
        return path

    monkeypatch.setattr(matpes.data, "download", fake_download)
    data, atoms = matpes.data.get_data("PBE", download_atoms=True, fmt=fmt)
    assert requested == [fmt, fmt]
    assert isinstance(data, list)
    assert len(atoms) == len(data) == len(records)
    for rec, got in zip(records, data, strict=True):
        assert Structure.from_dict(got["structure"]) == Structure.from_dict(rec["structure"])
        assert (
            Structure.from_dict(got["structure"]).site_properties
            == Structure.from_dict(rec["structure"]).site_properties
        )
        for key in (
            "matpes_id",
            "functional",
            "energy",
            "forces",
            "stress",
            "bandgap",
            "elements",
            "nelements",
            "chemsys",
            "formula_pretty",
            "formula_anonymous",
            "composition",
            "composition_reduced",
            "provenance",
            "cohesive_energy_per_atom",
            "formation_energy_per_atom",
            "bader_charges",
            "cm5_partial_charges",
        ):
            assert got.get(key) == rec.get(key), key
        assert got.get("ddec6") == rec.get("ddec6")
        np.testing.assert_allclose(got["abs_forces"], rec["abs_forces"])
        assert got["volume"] == pytest.approx(rec["volume"])
        assert got["density_atomic"] == pytest.approx(rec["density_atomic"])
    with pytest.raises(ValueError, match="Unknown format"):
        matpes.data.get_data(fmt="csv")  # type: ignore[arg-type]


def test_matgl(parquet_file, records, tmp_path):
    pytest.importorskip("matgl")
    from matgl.graph.data import MGLDiskDataset  # noqa: PLC0415

    from matpes.convert import to_mgl_dataset, write_mgl_shards  # noqa: PLC0415

    dataset = to_mgl_dataset(parquet_file, root=str(tmp_path / "mgl"), save_cache=False)
    assert len(dataset) == len(records)
    np.testing.assert_allclose(dataset.labels["stresses"][0], -0.1 * np.array(records[0]["stress"]))
    assert dataset.labels["energies"] == [r["energy"] for r in records]

    charged = to_mgl_dataset(parquet_file, include_charges=True, root=str(tmp_path / "mglq"), save_cache=False)
    assert len(charged) == sum(r.get("ddec6") is not None for r in records)

    root = write_mgl_shards(parquet_file, tmp_path / "shards", shard_size=3)
    disk = MGLDiskDataset(root)
    assert len(disk) == len(records)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {},
            [
                ("v2025.3", "pbe/train.parquet"),
                ("v2025.3", "pbe/validation.parquet"),
                ("v2025.3", "pbe/test.parquet"),
                ("v2025.3", "atoms/pbe.parquet"),
            ],
        ),
        ({"split": "train"}, [("v2025.3", "pbe/train.parquet"), ("v2025.3", "atoms/pbe.parquet")]),
        (
            {"version": "2025.2", "fmt": "jsonl"},
            [("v2025.2", "MatPES-PBE-2025.2.jsonl"), ("v2025.2", "MatPES-PBE-atoms.jsonl")],
        ),
        ({"version": "2025.2"}, [("v2025.2", "MatPES-PBE-2025.2.jsonl"), ("v2025.3", "atoms/pbe.parquet")]),
    ],
)
def test_get_data_downloads_one_format(monkeypatch, parquet_file, records, tmp_path, kwargs, expected):
    """Each dataset is fetched in one format only; Parquet is never paired with a JSONL download."""
    import matpes.data  # noqa: PLC0415

    fetched = []

    def fake_hf_hub_download(repo_id, filename, repo_type, revision=None, local_dir=None):
        fetched.append((revision, filename))
        dst = tmp_path / revision / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(parquet_file if filename.endswith(".parquet") else SAMPLE, dst)
        return str(dst)

    monkeypatch.setattr(matpes.data, "hf_hub_download", fake_hf_hub_download)
    data, _ = matpes.data.get_data("PBE", download_atoms=True, **kwargs)
    assert fetched == expected
    n_files = sum(not f.startswith(("atoms/", "MatPES-PBE-atoms")) for _, f in expected)
    assert len(data) == n_files * len(records)
    assert "structure" in data[0]
