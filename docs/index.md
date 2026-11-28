[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matpes)](https://github.com/materialsvirtuallab/matpes/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)

MatPES (Materials Potential Energy Surface) is a potential energy surface dataset with near-complete coverage of the
periodic table, designed to train foundation potentials (FPs), i.e., machine learning interatomic potentials (MLIPs)
for materials. MatPES is an initiative by the [Materialyze.AI lab] and the [Materials Project] to address
[critical deficiencies](http://matpes.ai/about) in existing PES datasets.

### Versions

| Version | Date | Description | Download |
| --- | --- | --- | --- |
| 2025.3 | 25 Nov 2026 | Removed 8,520 PBE and 7,514 r2SCAN duplicate records of rare earth compounds that are in the wrong low-spin state. Parquet with train/validation/test splits. | [PBE](https://huggingface.co/datasets/materialyze/matpes/tree/v2025.3/pbe), [r2SCAN](https://huggingface.co/datasets/materialyze/matpes/tree/v2025.3/r2scan) |
| 2025.2 | 15 Apr 2026 | Addition of Bader and DDEC6 charges; removed a small number of duplicated structures. | [PBE](https://huggingface.co/datasets/materialyze/matpes/resolve/v2025.2/MatPES-PBE-2025.2.jsonl?download=true), [r2SCAN](https://huggingface.co/datasets/materialyze/matpes/resolve/v2025.2/MatPES-R2SCAN-2025.2.jsonl?download=true) |
| 2025.1 | 6 Mar 2025 | Initial release (~400k structures) | [PBE](https://huggingface.co/datasets/materialyze/matpes/resolve/v2025.1/MatPES-PBE-2025.1.jsonl?download=true), [r2SCAN](https://huggingface.co/datasets/materialyze/matpes/resolve/v2025.1/MatPES-R2SCAN-2025.1.jsonl?download=true) |
| - | 6 Mar 2025 | Atomic reference energies | [PBE](https://huggingface.co/datasets/materialyze/matpes/resolve/v2025.3/atoms/pbe.parquet?download=true), [r2SCAN](https://huggingface.co/datasets/materialyze/matpes/resolve/v2025.3/atoms/r2scan.parquet?download=true) |


### Aims

1. **Accuracy.** MatPES is computed using static DFT calculations with stringent convergence criteria. Please refer
   to the `MatPESStaticSet` in [pymatgen] for details.
2. **Comprehensiveness.** MatPES structures are sampled using a 2-stage version of DImensionality-Reduced Encoded
   Clusters with sTratified ([DIRECT]) sampling from a greatly expanded configuration of MD structures.
3. **Quality.** MatPES includes computed data from the PBE functional, as well as the high fidelity r2SCAN meta-GGA
   functional with improved description across diverse bonding and chemistries.

MatPES 2025.3 is the latest public release. The initial 2025.1 release comprised ~400,000 structures from 300 K MD
simulations; 2025.2 added Bader and DDEC6 atomic charges and removed duplicated structures; 2025.3 removes 8,520 PBE
and 7,514 r2SCAN duplicate records of rare earth compounds that are in the wrong low-spin state. The dataset remains much
smaller than other PES datasets in the literature and yet achieves comparable or, in some cases,
[improved performance and reliability](http://matpes.ai/benchmarks) on trained FPs.

MatPES is part of the MatML ecosystem, which also includes [MatGL] (Materials Graph Library), [maml] (MAterials
Machine Learning), and [MatCalc] (Materials Calculator).

### Getting the Dataset

#### Hugging Face (Recommended)

The MatPES dataset is available on [Hugging Face](https://huggingface.co/datasets/materialyze/matpes). You can use
the `datasets` package to download it:

```python
from datasets import load_dataset

load_dataset("materialyze/matpes", "pbe")  # DatasetDict with train/validation/test splits

load_dataset("materialyze/matpes", "r2scan")

load_dataset("materialyze/matpes", "r2scan-atoms")  # Isolated-atom reference energies
```

The repository is organized as `pbe/{train,validation,test}.parquet`, `r2scan/{train,validation,test}.parquet` and
`atoms/{pbe,r2scan}.parquet`; the full dataset is the union of the three splits. `main` always holds the latest
release, and each release is a git tag. To pin a version, pass it as `revision`:

```python
load_dataset("materialyze/matpes", "pbe", revision="v2025.3")
load_dataset("materialyze/matpes", "pbe-2025.2", revision="v2025.2")  # JSONL releases use versioned configs.
```

#### Data formats

From 2025.3, MatPES is switching to [Parquet](https://parquet.apache.org/) as its distribution format. Earlier versions
remain available as JSONL. Compared with JSONL, Parquet offers:

- **Smaller downloads.** Columnar zstd compression makes the files about 4x smaller (e.g., 580 MB vs 2.3 GB for PBE).
- **Faster loading.** Numeric arrays are stored in binary form, so reading a full dataset is more than an order of
  magnitude faster than parsing JSON.
- **Read only what you need.** Column projection and predicate pushdown let you load selected fields or filter by,
  e.g., `chemsys` or `nsites` without reading the whole file.
- **Typed, lossless arrays.** A fixed schema with float64 throughout maps directly to zero-copy NumPy arrays.
- **Broad tooling support.** Parquet works out of the box with pandas, polars, pyarrow, DuckDB and Hugging Face
  `datasets`.

The two layouts are:

- **Parquet** (2025.3 onwards, `pbe/`, `r2scan/` and `atoms/` folders): flat, Arrow-native columns.
  The lattice is a fixed-size list of 9 floats, species a list of strings, positions/forces/magmoms per-atom lists,
  energy a scalar and stress a 6-vector, plus metadata (`functional`, `mp_id`, `formula_pretty`, `chemsys`,
  `bandgap`, Bader/CM5/DDEC6 charges, `provenance`). All floats are float64. See `matpes.parquet` for the full
  schema and units.
- **JSONL** (2025.1 and 2025.2, `MatPES-<PBE|R2SCAN>-<version>[-train|-valid|-test].jsonl`): one complete JSON record per line, including the serialized pymatgen `structure` and
  derived fields such as `composition` and `symmetry`.

`matpes` downloads Parquet by default and quietly falls back to JSONL for 2025.1 and 2025.2, which have no Parquet
files. `get_data()` always returns the familiar list of record dicts with `d["structure"]`; for fast columnar access,
use the Parquet files directly as below. `download()` returns a single file for one split, or the folder of all three
split files for the full dataset; every helper accepts either.

```python
import pyarrow.compute as pc
from matpes.data import download
from matpes.convert import to_structures, to_atoms, to_mgl_dataset, write_mgl_shards, to_lmdb
from matpes.parquet import read_table

path = download("R2SCAN", split="train")  # r2scan/train.parquet; download("R2SCAN") gives the full dataset

table = read_table(path, columns=["matpes_id", "energy", "nsites"], filters=pc.field("chemsys") == "Li-O")
structures = to_structures(path, filters=pc.field("chemsys") == "Li-O")  # pymatgen Structures
atoms = to_atoms(path, filters=[("nsites", "<", 8)])  # ASE Atoms with energy/forces/stress attached
dataset = to_mgl_dataset(path, filters=pc.field("chemsys") == "Li-O")  # MatGL MGLDataset
write_mgl_shards(path, "mgl_shards")  # Streamed on-disk MatGL shards for the full dataset
to_lmdb(path, "matpes-r2scan-train.lmdb")  # LMDB for random access; read with matpes.convert.MatPESLMDB
```

Every helper also accepts a JSONL file, so existing JSONL downloads keep working. Without any MatPES tooling:

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/materialyze/matpes/pbe/train.parquet", columns=["matpes_id", "chemsys", "energy"])
```

#### MatPES Package

The `matpes` python package, which provides tools for working with the MatPES datasets, can be installed via pip:

```shell
pip install matpes
```

Some command line usage examples:

```shell
# Download the PBE dataset (Parquet) into the Hugging Face cache and print its location.
matpes download pbe
matpes download r2scan --split train

# Extract all entries in the Fe-O chemical system from a Parquet file or folder (or JSONL; output format by suffix).
matpes data --chemsys Fe-O -o Fe-O.parquet path/to/pbe
matpes data --chemsys Fe-O -o Fe-O.jsonl MatPES-PBE-2025.2.jsonl

# Convert MatPES-style JSONL files (e.g., older versions or your own data) to Parquet.
matpes convert MatPES-PBE-2025.2.jsonl
```

The `ase`, `matgl` and `lmdb` integrations are optional: `pip install "matpes[ase,matgl,lmdb]"`.

The `matpes.db` module provides functionality to create your own MongoDB database with the downloaded MatPES data,
which is extremely useful if you plan to work with the data (e.g., querying, adding entries, etc.) extensively.

### MatPES-trained Models

We have released a set of MatPES-trained foundation potentials (FPs) in the [M3GNet], [CHGNet], and [TensorNet]
architectures in the [MatGL] package. For example, you can load the TensorNet FP trained on MatPES PBE 2025.2 as
follows:

```python
import matgl

potential = matgl.load_model("TensorNet-PES-MatPES-PBE-2025.2")
```

Model names follow the format `<architecture>-PES-<dataset>-<dataset-version>`.

These FPs can be used easily with the [MatCalc] package to rapidly compute properties. For example:

```python
from matcalc.elasticity import ElasticityCalc

calculator = ElasticityCalc("TensorNet-PES-MatPES-PBE-2025.2")
calculator.calc(structure)
```

### Tutorials

We have provided [Jupyter notebooks](http://matpes.ai/tutorials) demonstrating how to load the MatPES dataset, train
a model, and perform fine-tuning.

### Citing

If you use the MatPES dataset, please cite the following [work](https://doi.org/10.48550/arXiv.2503.04070):

```txt
Kaplan, A. D.; Liu, R.; Qi, J.; Ko, T. W.; Deng, B.; Riebesell, J.; Ceder, G.; Persson, K. A.; Ong, S. P. A
Foundational Potential Energy Surface Dataset for Materials. arXiv 2025. DOI: 10.48550/arXiv.2503.04070.
```

In addition, if you use any of the pre-trained FPs or architectures, please cite the
[references provided](http://matgl.ai/references) on the architecture used as well as MatGL.

[Materialyze.AI lab]: https://materialyze.ai
[Materials Project]: https://materialsproject.org
[M3GNet]: http://dx.doi.org/10.1038/s43588-022-00349-3
[CHGNet]: http://doi.org/10.1038/s42256-023-00716-3
[TensorNet]: https://arxiv.org/abs/2306.06482
[DIRECT]: https://doi.org/10.1038/s41524-024-01227-4
[pymatgen]: https://pymatgen.org
[maml]: https://materialsvirtuallab.github.io/maml/
[MatGL]: https://matgl.ai
[MatCalc]: https://matcalc.ai
