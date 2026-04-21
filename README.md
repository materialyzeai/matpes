[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matpes)](https://github.com/materialsvirtuallab/matpes/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matpes/workflows/Linting/badge.svg)

MatPES (Materials Potential Energy Surface) is a potential energy surface dataset with near-complete coverage of the
periodic table, designed to train foundation potentials (FPs), i.e., machine learning interatomic potentials (MLIPs)
for materials. MatPES is an initiative by the [Materialyze.AI lab] and the [Materials Project] to address
[critical deficiencies](http://matpes.ai/about) in existing PES datasets.

### Versions

| Version | Date | Description | Download |
|---------|------|-------------|----------|
| 2025.2 | 15 Apr 2026 | Addition of Bader and DDEC6 charges; removed a small number of duplicated structures. | [PBE](https://huggingface.co/datasets/materialyze/matpes/resolve/main/MatPES-PBE-2025.2-charges.json?download=true), [r2SCAN](https://huggingface.co/datasets/materialyze/matpes/resolve/main/MatPES-R2SCAN-2025.2-charges.json?download=true) |
| 2025.1 | 6 Mar 2025 | Initial release (~400k structures) | [PBE](https://huggingface.co/datasets/materialyze/matpes/resolve/main/MatPES-PBE-2025.1-charges.json?download=true), [r2SCAN](https://huggingface.co/datasets/materialyze/matpes/resolve/main/MatPES-R2SCAN-2025.1-charges.json?download=true) |
| -      | 6 Mar 2025 | Atomic reference energies | [PBE](https://huggingface.co/datasets/materialyze/matpes/resolve/main/MatPES-PBE-atoms.json?download=true), [r2SCAN](https://huggingface.co/datasets/materialyze/matpes/resolve/main/MatPES-R2SCAN-atoms.json?download=true) |

### Aims

1. **Accuracy.** MatPES is computed using static DFT calculations with stringent convergence criteria. Please refer
   to the `MatPESStaticSet` in [pymatgen] for details.
2. **Comprehensiveness.** MatPES structures are sampled using a 2-stage version of DImensionality-Reduced Encoded
   Clusters with sTratified ([DIRECT]) sampling from a greatly expanded configuration of MD structures.
3. **Quality.** MatPES includes computed data from the PBE functional, as well as the high fidelity r2SCAN meta-GGA
   functional with improved description across diverse bonding and chemistries.

The initial v2025.1 release comprises ~400,000 structures from 300K MD simulations. This dataset is much smaller
than other PES datasets in the literature and yet achieves comparable or, in some cases,
[improved performance and reliability](http://matpes.ai/benchmarks) on trained FPs.

MatPES is part of the MatML ecosystem, which also includes [MatGL] (Materials Graph Library), [maml] (MAterials
Machine Learning), and [MatCalc] (Materials Calculator).

### Getting the Dataset

#### Hugging Face (Recommended)

The MatPES dataset is available on [Hugging Face](https://huggingface.co/datasets/materialyze/matpes). You can use
the `datasets` package to download it:

```python
from datasets import load_dataset

load_dataset("materialyze/matpes", "pbe")

load_dataset("materialyze/matpes", "r2scan")
```

Without any version specifiers, the latest version of each dataset will be returned.

To download a specific version, append a `-<version>` specifier. For example:

```python
load_dataset("materialyze/matpes", "r2scan-2025.1")
```

#### MatPES Package

The `matpes` python package, which provides tools for working with the MatPES datasets, can be installed via pip:

```shell
pip install matpes
```

Some command line usage examples:

```shell
# Download the PBE dataset to the current directory.
# You should see a MatPES-PBE-2025.2.json file in your directory.
matpes download pbe

# Extract all entries in the Fe-O chemical system.
matpes data -i MatPES-PBE-2025.2.json --chemsys Fe-O -o Fe-O.json.gz
```

The `matpes.db` module provides functionality to create your own MongoDB database with the downloaded MatPES data,
which is extremely useful if you plan to work with the data (e.g., querying, adding entries, etc.) extensively.

### MatPES-trained Models

We have released a set of MatPES-trained foundation potentials (FPs) in the [M3GNet], [CHGNet], and [TensorNet]
architectures in the [MatGL] package. For example, you can load the TensorNet FP trained on MatPES PBE 2025.1 as
follows:

```python
import matgl

potential = matgl.load_model("TensorNet-MatPES-PBE-v2025.1-PES")
```

Model names follow the format `<architecture>-<dataset>-<dataset-version>-PES`.

These FPs can be used easily with the [MatCalc] package to rapidly compute properties. For example:

```python
from matcalc.elasticity import ElasticityCalc

calculator = ElasticityCalc("TensorNet-MatPES-PBE-v2025.1-PES")
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
