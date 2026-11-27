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
    ("2025.3", ("parquet", "jsonl")),
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

From 2025.3, each MatPES dataset is provided as a [Parquet](https://parquet.apache.org/) file with flat,
Arrow-native columns (float64 throughout): `lattice` (9 floats, row-major 3x3, Å), `pbc`, `species`, Cartesian
`positions` (Å), `forces` (eV/Å), `magmoms` (μB), `energy` (eV), `stress` (6-vector, VASP kbar), and metadata such as
`matpes_id`, `functional`, `mp_id`, `chemsys`, `bandgap`, Bader/CM5/DDEC6 charges and `provenance`. All versions are
also provided as [JSONL](https://jsonlines.org/) files, one JSON record per line. The atomic energies are computed with
the same settings as the datasets.

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

# Hugging Face datasets: "pbe-parquet" / "r2scan-parquet" have train, validation and test splits.
from datasets import load_dataset

ds = load_dataset("materialyze/matpes", "r2scan-parquet")

# Any Parquet reader, e.g., pandas, polars or pyarrow.
import pandas as pd

df = pd.read_parquet(path, columns=["matpes_id", "chemsys", "energy"])
```
"""

EXAMPLE_CONTENT = (
    """
### Example document

The following is a commented version of a single record in the JSONL layout (e.g., `MatPES-PBE-2025.3.jsonl`, or
`matpes.data.get_data(fmt="jsonl")`). The Parquet columns are flattened from these fields. Note that
the `bader_`, `cm5_partial_charges` and `ddec6` keys are None for structures where the charge calculations failed for
some reason.
These are a minority of structures.

```json
{
    "builder_meta": {  // Metadata used by MatPES build pipeline (emmet/pymatgen versions, timestamps).
        ...
    },

    "nsites": 2,  // Number of sites in the structure.
    "elements": ["Cd", "Sr"],  // Elements present in the structure.
    "nelements": 2,  // Number of unique elements in the structure.

    "composition": { "Sr": 1.0, "Cd": 1.0 },  // Elemental composition as a dictionary.
    "composition_reduced": { "Sr": 1.0, "Cd": 1.0 },  // Reduced/normalized composition.

    "formula_pretty": "SrCd",  // Readable chemical formula.
    "formula_anonymous": "AB",  // Anonymous formula representation.
    "chemsys": "Cd-Sr",  // Chemical system (elements sorted alphabetically, hyphen-separated).

    "volume": 69.60584595204928,  // Structure volume in ų.
    "density": 4.772002781547363,  // Density in g/cm³.
    "density_atomic": 34.80292297602464,  // Atomic density (volume per atom) in ų/atom.

    "symmetry": {  // Crystallographic symmetry information (via spglib).
        "crystal_system": "Triclinic",
        "symbol": "P1",
        "number": 1,
        "point_group": "1",
        "symprec": 0.1,
        "angle_tolerance": 5.0,
        "version": "2.5.0"
    },

    "structure": { ... },  // Pymatgen serialized Structure object (lattice, sites, species, coordinates).

    "energy": -34.92810022,  // DFT total energy in eV.

    "forces": [  // DFT-calculated forces on each atom (eV/Å), shape (nsites, 3).
        [0.28135608, 0.39456307, 0.28030103],
        [-0.28135608, -0.39456307, -0.28030103]
    ],

    "stress": [  // DFT-calculated stress tensor in Voigt notation (kbar): [xx, yy, zz, yz, xz, xy].
        -3.70051569, 45.57452896, -9.58430813,
        -0.80624803, 30.26332391, -29.08163294
    ],

    "matpes_id": "matpes-20240214_30496_38",  // Unique MatPES identifier for this structure.

    "bandgap": 0.0,  // DFT-calculated electronic band gap (eV).
    "functional": "r2SCAN",  // DFT exchange-correlation functional used.

    "formation_energy_per_atom": null,  // Formation energy per atom (eV). null if not a relaxed ground state.
    "cohesive_energy_per_atom": -1.2885743400000003,  // Cohesive energy per atom (eV).

    "abs_forces": [  // Magnitude of DFT force vector per atom (eV/Å).
        0.5598302665807309,
        0.5598302665807309
    ],

    "bader_charges": [8.867804, 13.132196],  // Bader-partitioned electron counts per atom (e). null if not computed.
    "bader_magmoms": [-4e-05, -2.4e-05],  // Bader-partitioned magnetic moments per atom (μ_B). null if not computed.

    "cm5_partial_charges": [-0.001647, 0.001647],  // CM5 partial atomic charges per atom (e), from Chargemol.

    "ddec6": {  // DDEC6 (Density Derived Electrostatic and Chemical) charge analysis results from Chargemol.
        "partial_charges": [0.803751, -0.803751],  // DDEC6 net atomic charges per atom (e).
        "bond_order_sums": [2.143492, 2.720226],  // Sum of DDEC6 bond orders for each atom.
        "spin_moments": [-5.4e-05, -1.1e-05],  // DDEC6 atomic spin moments per atom (μ_B).
        "dipoles": [  // DDEC6 atomic dipole vectors per atom (Debye), shape (nsites, 3).
            [-0.003428, -0.011809, -0.014139],
            [0.036623, 0.045952, -0.001567]
        ],
        "rsquared_moments": [42.045256, 61.998282],  // DDEC6 ⟨r²⟩ atomic multipole moments per atom (a.u.).
        "rcubed_moments": [107.916366, 168.708361],  // DDEC6 ⟨r³⟩ atomic multipole moments per atom (a.u.).
        "rfourth_moments": [368.510161, 601.960932]  // DDEC6 ⟨r⁴⟩ atomic multipole moments per atom (a.u.).
    },

    "provenance": {  // Metadata describing dataset origin and MD simulation conditions.
        "original_mp_id": "mp-30496",  // Source structure ID from the Materials Project.
        "materials_project_version": "v2022.10.28",  // Materials Project database version used.
        "md_ensemble": "NpT",  // Molecular dynamics ensemble type.
        "md_temperature": 300.0,  // MD simulation temperature (K).
        "md_pressure": 1.0,  // MD simulation pressure (atm).
        "md_step": 726,  // MD trajectory step number from which this snapshot was extracted.
        "mlip_name": "M3GNet-MP-2021.2.8-DIRECT"  // MLIP used to drive the MD trajectory.
    }
}
```
""",
)

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
