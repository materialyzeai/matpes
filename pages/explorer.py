"""Main MatPES Explorer App."""

from __future__ import annotations

import collections
import functools
import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.io as pio
from dash import Input, Output, State, callback, dcc, html
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
from pymatgen.core import Element

from matpes.db import MatPESDB

from .utils import pt_heatmap

if TYPE_CHECKING:
    import pandas as pd

dash.register_page(__name__, order=2)
DATADIR = Path(__file__).absolute().parent
# Define constants
FUNCTIONALS = ("PBE", "r2SCAN")

STATS = {}
for func in FUNCTIONALS:
    with open(DATADIR / f"{func.lower()}_stats.json") as f:
        STATS[func] = json.load(f)


DEFAULT_FIG_LAYOUT = dict(font=dict(size=18), yaxis=dict(nticks=8))


def get_data(
    functional: str,
    chemsys: str,
) -> pd.DataFrame:
    """
    Filter data.

    Args:
        functional (str): The functional used to filter the dataset (e.g., "PBE", "r2SCAN").
        chemsys (list of str): A list of chemical systems to filter by (e.g., ["Fe-O", "Ni-Mn"]).

    Returns:
        pd.DataFrame: Filtered data.
    """
    matpes = MatPESDB()
    return matpes.get_df(
        functional,
        criteria={"chemsys": chemsys},
        projection=[
            "formula_pretty",
            "elements",
            "cohesive_energy_per_atom",
            "abs_forces",
            "nsites",
            "nelements",
        ],
    )


def validate_chemsys(chemsys):
    """
    Validates and normalizes a chemical system string.

    This function checks whether the given chemical system string is valid and
    converts it into a normalized format where the chemical elements are sorted
    alphabetically. If the string is invalid, it returns None.

    Args:
        chemsys (str): A string representing a chemical system, where the elements
            are separated by a hyphen (e.g., "H-O-C").

    Returns:
        str or None: A normalized and sorted version of the chemical system string
            if valid. Returns None if the input is invalid.
    """
    try:
        toks = chemsys.split("-")
        for sym in toks:
            Element(sym)
        return "-".join(sorted(toks))
    except:  # noqa: E722
        pass
    return None


@callback(Output("chemsys_filter", "value"), Input("ptheatmap", "clickData"), State("chemsys_filter", "value"))
def update_chemsys_filter_on_click(clickdata, chemsys_filter):
    """
    Update chemsys_filter when PT table is clicked.

    Args:
        clickdata (dict): Click data.
        chemsys_filter (dict): Element filter.
    """
    new_chemsys_filter = chemsys_filter or ""
    chemsys = new_chemsys_filter.split("-")
    if clickdata:
        try:
            z = clickdata["points"][0]["text"].split("<")[0]
            chemsys.append(Element.from_Z(int(z)).symbol)
        except (ValueError, AttributeError):
            pass
    return "-".join(sorted(set(chemsys))).strip("-")


@callback(
    [
        Output("heatmap-title", "children"),
        Output("ptheatmap", "figure"),
        Output("coh_energy_hist", "figure"),
        Output("abs_forces_hist", "figure"),
        Output("nsites_hist", "figure"),
        Output("nelements_hist", "figure"),
        Output("data-div", "children"),
    ],
    [
        Input("functional", "value"),
        Input("chemsys_filter", "value"),
    ],
)
def display_data(
    functional,
    chemsys_filter,
):
    """
    Update graphs and data tables based on user-provided filters and criteria.

    This function processes the input filters and generates various visualizations and data structures,
    including a heatmap, histograms, and a formatted data table. The data is derived from a dataset
    filtered by the specified parameters.

    Args:
        functional (str): The functional used to filter the dataset (e.g., "PBE", "r2SCAN").
        chemsys_filter (list of str): A list of chemical systems to filter by (e.g., ["Fe-O", "Ni-Mn"]).

    Returns:
        tuple:
            - heatmap_figure (plotly.graph_objects.Figure): A heatmap of element counts, displayed in log scale.
            - histograms of formation energies, cohesive energies, nsites, nlements.
            - data table.
    """
    chemsys = validate_chemsys(chemsys_filter)
    return list(_build_outputs(functional, chemsys))


@functools.lru_cache(maxsize=512)
def _build_outputs(functional: str, chemsys: str | None) -> tuple:
    """
    Build the explorer outputs for a functional and (validated) chemical system. The underlying dataset is static, so
    results are cached to avoid repeated DB queries and figure construction.

    Args:
        functional (str): The functional (e.g., "PBE", "r2SCAN").
        chemsys (str | None): Normalized chemical system, or None for whole-dataset statistics.

    Returns:
        tuple: Outputs for the display_data callback.
    """
    df = None
    if chemsys:
        df = get_data(
            functional,
            chemsys,
        )
        data = {"nstructures": len(df)}
        data["element_counts"] = dict(collections.Counter(itertools.chain.from_iterable(df["elements"])))
        for c in ["cohesive_energy_per_atom", "nsites"]:
            counts, bins = np.histogram(df[c], bins=50)
            data[c] = {"counts": counts.tolist(), "bins": bins.tolist()}
        counts, bins = np.histogram(list(itertools.chain(*df["abs_forces"])), bins=50)
        data["abs_forces"] = {"counts": counts.tolist(), "bins": bins.tolist()}
        counts, bins = np.histogram(df["nelements"], bins=np.arange(0.5, 9.5, 1))

        data["nelements"] = {"counts": counts.tolist(), "bins": bins.tolist()}
    else:
        data = STATS[functional]
    nstructures = data["nstructures"]
    el_counts = collections.defaultdict(int)
    el_counts.update(data["element_counts"])

    def get_bin_mid(bins):
        bins = np.array(bins)
        return (bins[:-1] + bins[1:]) / 2

    current_template = pio.templates[pio.templates.default]
    colorway = current_template.layout.colorway
    ecoh_fig = px.bar(
        x=get_bin_mid(data["cohesive_energy_per_atom"]["bins"]),
        y=data["cohesive_energy_per_atom"]["counts"],
        labels={"x": "Cohesive Energy per Atom (eV/atom)", "y": "Count"},
        color_discrete_sequence=colorway,
    )
    ecoh_fig.update_layout(**DEFAULT_FIG_LAYOUT)
    forces_fig = px.bar(
        x=get_bin_mid(data["abs_forces"]["bins"]),
        y=data["abs_forces"]["counts"],
        labels={"x": "Absolute Forces (eV/A)", "y": "Count"},
        color_discrete_sequence=colorway[1:],
    )
    forces_fig.update_yaxes(title_text="Count", type="log")
    forces_fig.update_layout(showlegend=False, **DEFAULT_FIG_LAYOUT)

    nsites_fig = px.bar(
        x=get_bin_mid(data["nsites"]["bins"]),
        y=data["nsites"]["counts"],
        labels={"x": "nsites", "y": "Count"},
        color_discrete_sequence=colorway[2:],
    )

    nsites_fig.update_yaxes(title_text="Count", type="log")
    nsites_fig.update_layout(showlegend=False, **DEFAULT_FIG_LAYOUT)

    nelements_fig = px.bar(
        x=get_bin_mid(data["nelements"]["bins"]),
        y=data["nelements"]["counts"],
        labels={"x": "nelements", "y": "Count"},
        color_discrete_sequence=current_template.layout.colorway[3:],
    )
    nelements_fig.update_layout(showlegend=False, **DEFAULT_FIG_LAYOUT)

    output = [
        f"Elemental Heatmap ({nstructures:,} structures)",
        pt_heatmap(el_counts, label="Count", log=True, colorscale="YlOrRd_r"),
        ecoh_fig,
        forces_fig,
        nsites_fig,
        nelements_fig,
    ]

    if chemsys:
        table_df = df.drop("elements", axis=1)
        table_df = table_df.drop("abs_forces", axis=1)
        output.append(
            DataTable(
                page_size=25,
                id="data-table",
                columns=[
                    (
                        {
                            "name": i,
                            "id": i,
                            "type": "numeric",
                            "format": Format(precision=3, scheme=Scheme.fixed),
                        }
                        if i in ["energy", "cohesive_energy_per_atom"]
                        else {
                            "name": i,
                            "id": i,
                        }
                    )
                    for i in table_df.columns
                ],
                data=table_df.to_dict("records"),
            )
        )
    else:
        output.append("")
    return tuple(output)


# Define app layout
layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Functional"),
                        dcc.Dropdown(
                            id="functional",
                            options=[{"label": f, "value": f} for f in FUNCTIONALS],
                            value="PBE",
                            clearable=False,
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.Div("Chemical System"),
                        dcc.Input(
                            id="chemsys_filter",
                            placeholder="e.g., Li-Fe-O",
                            # Avoid a DB query on every keystroke while still filtering as the user types.
                            debounce=0.5,
                        ),
                    ],
                    width=2,
                ),
            ],
        ),
        dbc.Col(
            [
                html.Div(
                    "By default, statistics of the entire dataset are shown. Filtering by chemical system will "
                    "also display a table with basic information about the entries in that system. You can click on"
                    " the cells in the periodic table to set the chemical system as well. Filtering is done on the fly."
                ),
            ],
            width=12,
        ),
        dbc.Row(
            [
                html.H4("Elemental Heatmap", id="heatmap-title", className="section-title"),
                dbc.Col(
                    html.Div(
                        [dcc.Graph(id="ptheatmap")],
                        style={"marginLeft": "auto", "marginRight": "auto", "text-align": "center"},
                    ),
                    width=12,
                ),
            ]
        ),
        dbc.Row(
            [
                html.H4("Property Distribution", className="section-title"),
                dbc.Col(
                    dcc.Graph(
                        id="coh_energy_hist",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dcc.Graph(id="abs_forces_hist"),
                    width=6,
                ),
                dbc.Col(
                    dcc.Graph(id="nsites_hist"),
                    width=6,
                ),
                dbc.Col(
                    dcc.Graph(id="nelements_hist"),
                    width=6,
                ),
            ]
        ),
        html.Div(id="stats-div"),
        html.Div(id="data-div"),
    ]
)
