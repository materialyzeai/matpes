"""Benchmarks page."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from matpes import MATPES_SRC

dash.register_page(__name__, path="/dataset", order=4)

INTRO_CONTENT = f"""
### Introduction

Each MatPES dataset is provided as a gzipped file in the Javascript object notation (JSON) format. For example, the
`MatPES-PBE-2025.1.json.gz` file contains a list of structures with PES (energy, force, stresses) and associated
metadata. The [PBE]({MATPES_SRC}/MatPES-PBE-atoms.json) and [r2SCAN]({MATPES_SRC}/MatPES-R2SCAN-atoms.json)
atomic energies computed with the same  settings are also available. """

EXAMPLE_CONTENT = (
    """
### Example document

The following is a commented version of a single entry in the `MatPES-PBE-2025.1.json.gz` file. Note that the 
`bader_`, `cm5_partial_charges` and `ddec6` keys are None for structures where the charge calculations failed for some reason. 
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

    "formation_energy_per_atom": null,  // Formation energy per atom (eV). null if structure is not a relaxed ground state.
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
    f"""
#### Train-validation-test split

If you wish to reproduce the exact train:validation:test split used in the MatPES paper, you can download the split
indices for the [PBE]({MATPES_SRC}/MatPES-PBE-split.json.gz) and [r2SCAN]({MATPES_SRC}/MatPES-R2SCAN-split.json.gz).
You can then use the following code to split the dataset into train, validation, and test sets:
""",
    """
```python
from monty.serialization import loadfn
import json

pbe = loadfn("MatPES-PBE-2025.1.json.gz")
splits = loadfn("MatPES-PBE-split.json.gz")

train_set = []
valid_set = []
test_set = []

for i, d in enumerate(pbe):
    if i in splits["train"]:
        train_set.append(d)
    elif i in splits["valid"]:
        valid_set.append(d)
    else:
        test_set.append(d)

print(f"{len(train_set)}-{len(valid_set)}-{len(test_set)}")
# Output is 391240-21735-21737
```

""",
)

layout = dbc.Container([html.Div([dcc.Markdown(INTRO_CONTENT), dcc.Markdown(EXAMPLE_CONTENT)])])
