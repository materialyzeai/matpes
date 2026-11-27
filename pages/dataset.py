"""Dataset page."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from matpes import MATPES_SRC

dash.register_page(__name__, path="/dataset", order=4)

FMT_LABELS = {"parquet": "Parquet", "jsonl": "JSONL"}


def _link(name: str, label: str) -> str:
    return f"[{label}]({MATPES_SRC}/{name}?download=true)"


def _files(functional: str, version: str, fmts: tuple[str, ...]) -> str:
    return ", ".join(_link(f"MatPES-{functional}-{version}.{fmt}", FMT_LABELS[fmt]) for fmt in fmts)


def _splits(functional: str, version: str, fmt: str) -> str:
    return " / ".join(_link(f"MatPES-{functional}-{version}-{s}.{fmt}", s) for s in ("train", "valid", "test"))


DOWNLOADS = [
    ("2025.3", ("parquet",)),
    ("2025.2", ("jsonl",)),
    ("2025.1", ("jsonl",)),
]

DOWNLOAD_TABLE = "\n".join(
    [
        "| Version | PBE | r2SCAN | Train / valid / test splits |",
        "| --- | --- | --- | --- |",
        *(
            f"| {version} | {_files('PBE', version, fmts)} | {_files('R2SCAN', version, fmts)} | "
            + (
                f"PBE {_splits('PBE', version, fmts[0])}; r2SCAN {_splits('R2SCAN', version, fmts[0])}"
                if version != "2025.1"
                else "-"
            )
            + " |"
            for version, fmts in DOWNLOADS
        ),
        (
            f"| Atomic energies | {_files('PBE', 'atoms', ('parquet', 'jsonl'))} | "
            f"{_files('R2SCAN', 'atoms', ('parquet', 'jsonl'))} | - |"
        ),
    ]
)

INTRO_CONTENT = """
### Introduction

From 2025.3, MatPES is switching to the [Parquet](https://parquet.apache.org/) format. Compared with the JSONL files
used for earlier versions, Parquet offers:

- **Smaller downloads.** Columnar zstd compression makes the files about 4x smaller (e.g., 565 MB vs 2.3 GB for PBE).
- **Faster loading.** Numeric arrays are stored in binary form, so reading a full dataset is more than an order of
  magnitude faster than parsing JSON.
- **Read only what you need.** Column projection and predicate pushdown let you load selected fields or filter by,
  e.g., `chemsys` or `nsites` without reading the whole file.
- **Typed, lossless arrays.** A fixed schema with float64 throughout maps directly to zero-copy NumPy arrays.
- **Broad tooling support.** Parquet works out of the box with pandas, polars, pyarrow, DuckDB and Hugging Face
  `datasets`.

Each MatPES dataset is provided as a Parquet file with flat, Arrow-native columns (float64 throughout): `lattice` (9
floats, row-major 3x3, Å), `pbc`, `species`, Cartesian `positions` (Å), `forces` (eV/Å), `magmoms` (μB), `energy`
(eV), `stress` (6-vector, VASP kbar), and metadata such as `matpes_id`, `functional`, `mp_id`, `chemsys`, `bandgap`,
Bader/CM5/DDEC6 charges and `provenance`. Versions 2025.1 and 2025.2 are provided as [JSONL](https://jsonlines.org/)
files, one JSON record per line. The atomic energies are computed with the same settings as the datasets.

### Downloads

All files are hosted on [Hugging Face](https://huggingface.co/datasets/materialyze/matpes). Splits are a random
90/5/5 train/valid/test partition.
"""

LOADING_CONTENT = """
### Loading the data

```python
# matpes package: Parquet by default, JSONL for versions without Parquet files.
from matpes.data import download, get_data
from matpes.convert import to_structures

data = get_data("R2SCAN")  # List of record dicts, each with d["structure"].
path = download("R2SCAN", split="train")  # MatPES-R2SCAN-2025.3-train.parquet
structures = to_structures(path)  # Also to_atoms, to_mgl_dataset, write_mgl_shards and to_lmdb.

# Hugging Face datasets: "pbe-split" / "r2scan-split" have train, validation and test splits.
from datasets import load_dataset

ds = load_dataset("materialyze/matpes", "r2scan-split")

# Any Parquet reader, e.g., pandas, polars or pyarrow.
import pandas as pd

df = pd.read_parquet(path, columns=["matpes_id", "chemsys", "energy"])
```
"""

EXAMPLE_CONTENT = """
### Example row

The following is a commented version of a single row of `MatPES-PBE-2025.3.parquet`, e.g., as returned by
`pyarrow.parquet.read_table(path, filters=[("matpes_id", "==", "matpes-20240214_30496_38")]).to_pylist()[0]`. Each
comment gives the Arrow type and units. The Bader, CM5 and DDEC6 columns are null for the minority of structures where
the charge calculations failed.

```json
{
    "matpes_id": "matpes-20240214_30496_38",  // string. Unique MatPES identifier.
    "functional": "PBE",  // string. "PBE" or "r2SCAN".
    "mp_id": "mp-30496",  // string. Source Materials Project ID.
    "formula_pretty": "SrCd",  // string. Reduced formula.
    "chemsys": "Cd-Sr",  // string. Elements sorted alphabetically, hyphen-separated.
    "nsites": 2,  // int32. Number of sites.

    "lattice": [  // fixed_size_list<float64, 9>. Row-major 3x3 lattice matrix (Å).
        4.680644111000686, 0.0, 0.0,
        -0.6449861383871763, 3.1871035434547226, 0.0,
        -0.3884423008526454, -0.2858736045691578, 4.665991267765886
    ],
    "pbc": [true, true, true],  // fixed_size_list<bool, 3>. Periodic boundary conditions.
    "species": ["Sr", "Cd"],  // list<string>. Element of each site.
    "positions": [  // list<fixed_size_list<float64, 3>>. Cartesian coordinates (Å), shape (nsites, 3).
        [4.007651111000689, 1.0898900000000027, 1.9904600000000021],
        [1.5305876991473655, 0.42296139543083866, 4.372134267765877]
    ],

    "energy": -2.13697943,  // float64. DFT total energy (eV).
    "forces": [  // list<fixed_size_list<float64, 3>>. DFT forces (eV/Å), shape (nsites, 3).
        [0.27601894, 0.36633401, 0.27819465],
        [-0.27601894, -0.36633401, -0.27819465]
    ],
    "stress": [  // fixed_size_list<float64, 6>. DFT stress (kbar, VASP sign convention, compressive positive),
        4.93532684, 54.77972399, 0.01626795,  // Voigt order [xx, yy, zz, yz, xz, xy].
        -0.51256794, 29.10553507, -27.91916545
    ],
    "magmoms": [-0.0, 0.0],  // list<float64>. VASP site magnetic moments (μB).

    "bandgap": 0.0,  // float64. DFT band gap (eV).
    "cohesive_energy_per_atom": -1.047500055,  // float64. Cohesive energy (eV/atom).
    "formation_energy_per_atom": null,  // float64. Formation energy (eV/atom); null unless a relaxed ground state.

    "bader_charges": [9.001567, 12.998434],  // list<float64>. Bader electron counts per atom (e).
    "bader_magmoms": [-0.00026, 0.000135],  // list<float64>. Bader magnetic moments per atom (μB).
    "cm5_partial_charges": [-0.027626, 0.027626],  // list<float64>. CM5 partial charges per atom (e).
    "ddec6_partial_charges": [0.703964, -0.703964],  // list<float64>. DDEC6 net atomic charges (e).
    "ddec6_spin_moments": [-0.000293, 0.000168],  // list<float64>. DDEC6 atomic spin moments (μB).
    "ddec6_bond_order_sums": [2.208307, 2.641683],  // list<float64>. Sum of DDEC6 bond orders per atom.
    "ddec6_dipoles": [  // list<fixed_size_list<float64, 3>>. DDEC6 atomic dipoles, shape (nsites, 3).
        [-0.000783, -0.010785, -0.011862],
        [0.026313, 0.04049, -0.003057]
    ],
    "ddec6_rsquared_moments": [43.276448, 60.520927],  // list<float64>. DDEC6 ⟨r²⟩ moments (a.u.).
    "ddec6_rcubed_moments": [112.814581, 162.315746],  // list<float64>. DDEC6 ⟨r³⟩ moments (a.u.).
    "ddec6_rfourth_moments": [389.37931, 572.965689],  // list<float64>. DDEC6 ⟨r⁴⟩ moments (a.u.).

    "provenance": {  // struct. Origin of the structure and MD sampling conditions.
        "original_mp_id": "mp-30496",  // string. Source Materials Project ID.
        "materials_project_version": "v2022.10.28",  // string. Materials Project database version.
        "md_ensemble": "NpT",  // string. MD ensemble.
        "md_temperature": 300.0,  // float64. MD temperature (K).
        "md_pressure": 1.0,  // float64. MD pressure (atm).
        "md_step": 726,  // int64. MD step from which this snapshot was taken.
        "mlip_name": "M3GNet-MP-2021.2.8-DIRECT"  // string. MLIP used to drive the MD trajectory.
    }
}
```

The JSONL records of 2025.1 and 2025.2 hold the same data plus fields derivable from it: a serialized pymatgen
`structure`, `elements`, `composition`, `formula_anonymous`, `volume`, `density`, `symmetry`, `abs_forces` and a
nested `ddec6` dict. `matpes.data.get_data()` returns records in this JSONL layout for every version, rebuilding the
`structure` and composition fields from the Parquet columns.
"""

layout = dbc.Container(
    [
        html.Div(
            [
                dcc.Markdown(INTRO_CONTENT),
                dcc.Markdown(DOWNLOAD_TABLE, id="downloads-table"),
                dcc.Markdown(LOADING_CONTENT),
                dcc.Markdown(EXAMPLE_CONTENT),
            ]
        )
    ]
)
