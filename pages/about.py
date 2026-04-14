"""Home page."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from matpes import MATPES_SRC

dash.register_page(__name__, path="/about", order=6)

MARKDOWN_CONTENT = f"""
### Background

Machine learning interatomic potentials (MLIPs) have revolutionized the field of computational materials science.
MLIPs use ML to reproduce the PES (energies, forces, and stresses) of a collection of atoms, typically computed
using an ab initio method such as density functional theory (DFT).
This enables the simulation of materials at much larger length and longer time scales at near-ab initio accuracy.

One of the most exciting developments in the past few years is the emergence of MLIPs with near-complete coverage of
the periodic table of elements. Such universal MLIPs are also known as foundation potentials (FPs). Examples include
[M3GNet], [CHGNet], [MACE], to name a few. FPs have broad applications, including materials discovery and the
prediction of PES-derived properties such as elastic constants, phonon dispersion, etc.

However, most current FPs were trained on DFT relaxation calculations, e.g., from the [Materials Project].
This dataset, referred to as `MPF` or `MPTraj` in the literature, suffer from several issues:

1. The energies, forces, and stresses are not converged to the accuracies necessary to train a high quality MLIP.
2. Most of the structures are near-equilibrium, with very little coverage of non-equilibrium local environments.
3. The calculations were performed using the common Perdew-Burke-Ernzerhof (PBE) generalized gradient approximation
   (GGA) functional, even though improved functionals with better performance across diverse chemistries and bonding
   such as the strongly constrained and appropriately normed (SCAN) meta-GGA functional already exists.

### Goals

MatPES is an initiative by the [Materialyze.AI lab] and the [Materials Project] to address these limitations
comprehensively. The aims of MatPES are three-fold:

1. **Accuracy.** The data in MatPES was computed using static DFT calculations with stringent converegence criteria.
   Please refer to the `MatPESStaticSet` in [pymatgen] for details.
2. **Comprehensiveness.** The structures in MatPES are using a 2-stage version of DImensionality-Reduced
   Encoded Clusters with sTratified ([DIRECT]) sampling from a greatly expanded configuration of structures from MD
   simulations with the pre-trained [M3GNet] FP.
3. **Quality.** MatPES contains not only data computed using the PBE functional, but also the revised regularized SCAN
   (r2SCAN) meta-GGA functional. The r2SCAN functional recovers all 17 exact constraints presently known for
   meta-GGA functionals and has shown good transferable accuracy across diverse bonding and chemistries.

The workflow used to generate the MatPES dataset is shown below. The initial v2025 release comprises ~400,000
structures from 300K MD simulations and Materials Project ground state calculations. This dataset is much smaller
than other PES datasets in the literature and yet achieves essentially comparable or, in some cases, improved
performance and reliability. The [MatPES.ai] website provides a comprehensive analysis of the
[statistics](http://matpes.ai/explorer) and [performance benchmarks](http://matpes.ai/benchmarks) of the MatPES
dataset.

![matpes_workflow](assets/MatPES_workflow.png)

### Citing MatPES

If you use MatPES, please cite the following work:

```txt
Aaron Kaplan, Runze Liu, Ji Qi, Tsz Wai Ko, Bowen Deng, Gerbrand Ceder, Kristin A. Persson, Shyue Ping Ong.
A foundational potential energy surface dataset for materials. Submitted.
```

[Materialyze.AI lab]: https://materialyze.ai
[pymatgen]: https://pymatgen.org
[Materials Project]: https://materialsproject.org
[MatGL]: https://matgl.ai
[M3GNet]: http://dx.doi.org/10.1038/s43588-022-00349-3
[CHGNet]: http://doi.org/10.1038/s42256-023-00716-3
[MACE]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html
[DIRECT]: https//doi.org/10.1038/s41524-024-01227-4
[MatPES.ai]: https://matpes.ai

"""

layout = dbc.Container([html.Div([dcc.Markdown(MARKDOWN_CONTENT)])])
