"""Benchmarks page."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

dash.register_page(__name__, path="/benchmarks", order=3)

DATADIR = Path(__file__).absolute().parent

BENCHMARK_DATA = {func: pd.read_csv(DATADIR / f"matcalc-benchmark-{func.lower()}.csv") for func in ("PBE", "r2SCAN")}


INTRO_CONTENT = """
## MatCalc-Benchmark

The MatCalc-Benchmark evaluates FP performance across equilibrium, near-equilibrium, and molecular dynamics
properties.

Important notes:
- Beyond property metrics, training data size and model complexity are crucial.
- Large datasets demand significant computational resources. TensorNet training on MatPES-PBE takes ~15 min/epoch on an
  RTX A6000 GPU, whereas OMat24 requires ~20 hours/epoch on 16 A100 GPUs.
- Complex models with high parameter counts are computationally expensive, restricting MD simulation scale and time.
  For instance, eqV2-OMat24 has a t_step of ~213 ms/atom/step, nearly 100 times costlier than the models reported here.
- Performance differences should be viewed in context of statistical significance. Given that the same datasets are
  used for all models, statistical significance is determined using a
  [paired t-test](https://en.wikipedia.org/wiki/Paired_difference_test) at alpha=0.05.

We welcome the community's contribution of FPs to this MatCalc-Benchmark. To ensure a fair
comparison, please provide **information about training dataset size, training cost, and the number of parameters**.
The easiest way to run the benchmark is to implement an ASE compatible calculator, which can then be used with the
[MatCalc](https://github.com/materialsvirtuallab/matcalc) package. We will release the equilibrium and
near-equilibrium benchmark datasets soon in the MatCalc repository together with benchmarking tools. The MD
benchmarks can only be run by the [Materialyze.AI lab](https://materialyze.ai).
"""

TABLE_NOTE = """
For MAEs, all values not statistically different from the best value in each column are highlighted. Statistical
significance is determined using a [paired t-test](https://en.wikipedia.org/wiki/Paired_difference_test) with 0.05
alpha level. It should be noted that the Ef test set was derived from the WBM database, which was computed using a
different set of pseudopotential settings. This is likely the reason why the reported Ef MAEs are higher than
expected. We are in the process of updating the Ef benchmark data with consistent settings for the MatPES models. We
expect the performance to be comparable to the OMat24 models.
"""

LEGEND = r"""
##### Metrics

MatCalc-Benchmark metrics can be divided into three categories: equilibrium, near-equilibrium, and molecular dynamics
properties.

| Task                          | Symbol     | Units        | Functional   | Test Source              | Number |
|-------------------------------|------------|--------------|--------------|--------------------------|--------|
| **Equilibrium**               |            |              |              |                          |        |
| Structural similarity         | d          | -            | PBE          | [WBM]                    | 1,000  |
|                               |            | -            | r2SCAN       | [GNoME]                  | 1,000  |
| Formation energy per atom     | Ef         | eV/atom      | PBE          | [WBM]                    | 1,000  |
|                               |            | eV/atom      | r2SCAN       | [GNoME]                  | 1,000  |
| **Near-equilibrium**          |            |              |              |                          |        |
| Bulk modulus                  | K_VRH      | GPa          | PBE          | [MP]                     | 3,959  |
| Shear modulus                 | G_VRH      | GPa          | PBE          | [MP]                     | 3,959  |
| Constant volume heat capacity | C_V        | J/mol/K      | PBE          | [Alexandria]             | 1,170  |
| Off-equilibrium force         | F/F_DFT    | --           | PBE          | [WBM high energy states] | 979    |
| **Molecular dynamics**        |            |              |              |                          |        |
| Median termination temp       | T_1/2^term | K            | PBE & r2SCAN | [Materialyze.AI lab]     | 172    |
| Ionic conductivity            | sigma      | mS/cm        | PBE          | [Materialyze.AI lab]     | 698    |
| Time per atom per step        | t_step     | ms/step/atom | PBE & r2SCAN | [Materialyze.AI lab]     | 1      |

The time per atom per step (t_step) was computed using LAMMPS MD simulations conducted on a single Intel Xeon Gold core
for a system of 64 Si atoms under ambient conditions (300 K and 1 bar) over 50 ps with a 1 fs time step.

##### Datasets

The current MatCalc-Benchmark includes M3GNet, CHGNet and TensorNet FPs trained on the MatPES, MPF,
MPtrj, and OMat24 datasets, summarized below.

| Dataset       | Number of Structures |
|--------------|---------------------|
| MPF         | 185,877             |
| MPtrj       | 1,580,395           |
| OMat24      | 100,824,585         |
| MatPES PBE  | 434,712             |
| MatPES r²SCAN | 387,897           |

[WBM]: https://doi.org/10.1038/s41524-020-00481-6
[GNoME]: https://doi.org/10.1038/s41586-023-06735-9
[Alexandria]: https://doi.org/10.48550/arXiv.2412.16551
[WBM high energy states]: https://doi.org/10.48550/arXiv.2405.07105
[MP]: http://materialsproject.org
[Materialyze.AI lab]: https://materialyze.ai
"""


def get_sorted(df, i):
    """
    Determine the best value from a specified column in a DataFrame.

    This function selects either the maximum or minimum value of a specified column
    based on the input. For specific column names, the maximum value is chosen,
    while for all other columns, the minimum value is selected.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to evaluate.
        i (str): The name of the column to determine the best value from.

    Returns:
        Sorted list of values from the specified column.
    """
    if i in ("f_FP/f_DFT", "T_1/2^term (K)"):
        return sorted(df[i].dropna(), reverse=True)
    return sorted(df[i].dropna())


@callback(
    [Output("pbe-graph", "figure"), Output("r2scan-graph", "figure")],
    [
        Input("pbe-benchmarks-table", "selected_columns"),
        Input("r2scan-benchmarks-table", "selected_columns"),
        Input("pbe-benchmarks-table", "selected_rows"),
        Input("r2scan-benchmarks-table", "selected_rows"),
    ],
)
def update_graphs(selected_columns_pbe, selected_columns_r2scan, selected_rows_pbe, selected_rows_r2scan):
    """

    @callback(
        [Output("pbe-graph", "figure"), Output("r2scan-graph", "figure")],
        [
            Input("pbe-benchmarks-table", "selected_columns"),
            Input("r2scan-benchmarks-table", "selected_columns"),
            Input("pbe-benchmarks-table", "selected_rows"),
            Input("r2scan-benchmarks-table", "selected_rows"),
        ],
    ).

    Update the graphs based on the selected columns and rows for PBE and R2SCAN benchmarks.

    Parameters:
    - selected_columns_pbe: List of selected columns for PBE benchmarks.
    - selected_columns_r2scan: List of selected columns for R2SCAN benchmarks.
    - selected_rows_pbe: List of selected rows for PBE benchmarks.
    - selected_rows_r2scan: List of selected rows for R2SCAN benchmarks.

    Returns:
    A list of figures updated based on the selected columns and rows for PBE and R2SCAN benchmarks.

    """
    layout = dict(font=dict(size=18))
    figs = []
    for cols, rows, (_func, df) in zip(
        [selected_columns_pbe, selected_columns_r2scan],
        [selected_rows_pbe, selected_rows_r2scan],
        BENCHMARK_DATA.items(),
        strict=False,
    ):
        to_plot = df.iloc[rows]
        col = cols[0]
        fig = px.bar(to_plot, x="Dataset", y=col, color="Architecture", barmode="group")
        fig.update_layout(**layout)
        figs.append(fig)
    return figs


def gen_data_table(df, name):
    """
    Generates a Dash DataTable with specific configurations for displaying benchmarking
    data from a Pandas DataFrame. The table filters out certain columns, formats numeric
    data, and applies conditional styling to rows and columns based on specified criteria.

    Parameters:
        df (pd.DataFrame): The Pandas DataFrame containing data to display in the table.
        Columns in the DataFrame will be filtered and styled based on the function's logic.

    Returns:
        dash.dash_table.DataTable: A Dash DataTable object configured with the data, styling,
        and sorting properties derived from the input DataFrame.
    """
    cols = [c for c in df.columns if c if not ("diff" in c or "STDAE" in c)]
    return dash_table.DataTable(
        id=f"{name}-benchmarks-table",
        columns=[
            {
                "name": i,
                "id": i,
                "type": "numeric",
                "selectable": i not in ["Dataset", "Architecture"],
                "format": Format(precision=2, scheme=Scheme.decimal, nully="-"),
            }
            for i in cols
        ],
        data=df.to_dict("records"),
        column_selectable="single",
        row_selectable="multi",
        selected_columns=["d MAE"],
        selected_rows=list(range(len(df))),
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "rgb(220, 220, 220)",
            }
        ]
        + [
            {
                "if": {"column_id": i, "row_index": np.where(~df[f"{i.split(' ')[0]} sig_diff_rel"])[0]},
                "font-weight": "bold",
                "color": "white",
                "background-color": "#633D9Caa",
            }
            for i in cols
            if i.endswith("MAE")
        ]
        + [
            {
                "if": {
                    "filter_query": "{{T_1/2^term}} = {}".format(df["T_1/2^term"].max()),
                    "column_id": "T_1/2^term",
                },
                "font-weight": "bold",
                "color": "white",
                "background-color": "#633D9Caa",
            },
            {
                "if": {
                    "filter_query": "{{f/f_DFT}} = {}".format(df["f/f_DFT"].max() if "f/f_DFT" in df else 0),
                    "column_id": "f/f_DFT",
                },
                "font-weight": "bold",
                "color": "white",
                "background-color": "#633D9Caa",
            },
            {
                "if": {
                    "filter_query": "{{t_step}} = {}".format(df["t_step"].min()),
                    "column_id": "t_step",
                },
                "font-weight": "bold",
                "color": "white",
                "background-color": "#633D9Caa",
            },
        ],
        style_header={"backgroundColor": "#633D9C", "color": "white", "fontWeight": "bold"},
        sort_action="native",
    )


pbe_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Col(dcc.Graph(id="pbe-graph"), width=12),
            dbc.Col(
                html.Div(
                    "Clicking on the radio buttons graphs the selected column.",
                ),
                width=12,
            ),
            dbc.Col(
                gen_data_table(BENCHMARK_DATA["PBE"], "pbe"),
                width=12,
            ),
            dbc.Col(
                html.Div(dcc.Markdown(TABLE_NOTE)),
                width=12,
            ),
        ]
    ),
    className="mt-3",
)

r2scan_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Col(
                html.Div(
                    "There are only a limited number of MatPES r2SCAN benchmarks for different models due to"
                    " the limited amount of other r2SCAN training data sets and ground-truth r2SCAN DFT data.",
                ),
                width=12,
            ),
            dbc.Col(dcc.Graph(id="r2scan-graph"), width=12),
            dbc.Col(
                html.Div(
                    "Clicking on the radio buttons graphs the selected column.",
                ),
                width=12,
            ),
            dbc.Col(
                gen_data_table(BENCHMARK_DATA["r2SCAN"], "r2scan"),
                width=12,
            ),
            dbc.Col(
                html.Div(dcc.Markdown(TABLE_NOTE)),
                width=12,
            ),
        ]
    ),
    className="mt-3",
)


layout = dbc.Container(
    [
        dbc.Col(
            html.Div([dcc.Markdown(INTRO_CONTENT)]),
            width=12,
        ),
        dbc.Tabs(
            [
                dbc.Tab(pbe_tab, label="PBE"),
                dbc.Tab(r2scan_tab, label="r2SCAN"),
            ]
        ),
        dbc.Col(html.H4("Additional Information"), width=12, style={"padding-top": "20px"}),
        dbc.Col(
            html.Div([dcc.Markdown(LEGEND)], id="matcalc-benchmark-legend"),
            width=12,
            style={"padding-top": "10px"},
        ),
    ]
)
