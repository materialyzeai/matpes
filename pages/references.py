"""Home page."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

dash.register_page(__name__, path="/references", order=7)

MARKDOWN_CONTENT = """
#### References

[MatPES](https://doi.org/10.48550/arXiv.2503.04070)

```txt
Kaplan, A. D.; Liu, R.; Qi, J.; Ko, T. W.; Deng, B.; Riebesell, J.; Ceder, G.; Persson, K. A.; Ong, S. P. A
Foundational Potential Energy Surface Dataset for Materials. arXiv 2025. DOI: 10.48550/arXiv.2503.04070
```

[M3GNet]

```txt
Chen, C.; Ong, S. P. A Universal Graph Deep Learning Interatomic Potential for the Periodic Table. Nat Comput
Sci 2022, 2 (11), 718-728. DOI: 10.1038/s43588-022-00349-3
```

[CHGNet]

```txt
Deng, B.; Zhong, P.; Jun, K.; Riebesell, J.; Han, K.; Bartel, C. J.; Ceder, G. CHGNet as a Pretrained Universal
Neural Network Potential for Charge-Informed Atomistic Modelling. Nat Mach Intell 2023, 5 (9), 1031-1041.
DOI: 10.1038/s42256-023-00716-3.
```

[TensorNet]

```txt
Simeon, G.; de Fabritiis, G. TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
Potentials. arXiv October 30, 2023. DOI: 10.48550/arXiv.2306.06482.
```

[MatGL]

```txt
Ko, T. W.; Deng, B.; Nassar, M.; Barroso-Luque, L.; Liu, R.; Qi, J.; Liu, E.; Ceder, G.; Miret, S.;
Ong, S. P. Materials Graph Library (MatGL), an open-source graph deep learning library for materials science and
chemistry. Submitted.
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
