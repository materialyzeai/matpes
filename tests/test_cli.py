from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from matpes.cli import main

SAMPLE = Path(__file__).parent / "files" / "MatPES-PBE-sample.jsonl"


def _run(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["matpes", *map(str, args)])
    main()


@pytest.fixture
def records() -> list[dict]:
    with SAMPLE.open() as f:
        return [json.loads(line) for line in f]


def test_convert(monkeypatch, tmp_path, records):
    out = tmp_path / "out.parquet"
    _run(monkeypatch, "convert", SAMPLE, "-o", out, "-v", "2025.3")
    assert pq.read_metadata(out).num_rows == len(records)


@pytest.mark.parametrize("out_suffix", [".jsonl", ".parquet"])
def test_data_filter_jsonl(monkeypatch, tmp_path, records, out_suffix):
    chemsys = sorted({r["chemsys"] for r in records})[:2]
    out = tmp_path / f"filtered{out_suffix}"
    _run(monkeypatch, "data", "-c", *chemsys, "-o", out, SAMPLE)
    expected = [r["matpes_id"] for r in records if r["chemsys"] in chemsys]
    if out_suffix == ".parquet":
        got = pq.read_table(out).column("matpes_id").to_pylist()
    else:
        got = [json.loads(line)["matpes_id"] for line in out.open()]
    assert got == expected
    assert len(expected) >= 2


def test_data_filter_parquet(monkeypatch, tmp_path, records):
    src = tmp_path / "all.parquet"
    _run(monkeypatch, "convert", SAMPLE, "-o", src)
    formula = records[0]["formula_pretty"]
    out = tmp_path / "filtered.parquet"
    _run(monkeypatch, "data", "-f", formula, "-o", out, src)
    expected = [r["matpes_id"] for r in records if r["formula_pretty"] == formula]
    assert pq.read_table(out).column("matpes_id").to_pylist() == expected

    with pytest.raises(SystemExit, match="Parquet output"):
        _run(monkeypatch, "data", "-f", formula, "-o", tmp_path / "x.jsonl", src)
