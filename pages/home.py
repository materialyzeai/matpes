"""Home page."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from matpes import MATPES_SRC

dash.register_page(__name__, path="/", order=1)

readme = Path(__file__).parent.absolute() / ".." / "README.md"

with open(readme, encoding="utf-8") as f:
    MARKDOWN_CONTENT = f.read()

MARKDOWN_CONTENT = "\n".join(MARKDOWN_CONTENT.split("\n")[2:])

jumbotron = html.Div(
    dbc.Container(
        [
            html.H1("MatPES", className="display-3", id="matpes-title"),
            html.P(
                "A Foundational Potential Energy Surface Dataset for Materials.",
                className="lead",
            ),
            html.Hr(className="my-2"),
            dbc.Row(
                html.Div(
                    [
                        dbc.Button(
                            "PBE",
                            href=f"{MATPES_SRC}/MatPES-PBE-2025.2-charges.json?download=true",
                            class_name="me-1 download-button",
                            color="info",
                            external_link=True,
                            size="lg",
                            id="pbe-download-button",
                        ),
                        dbc.Tooltip(
                            "Download PBE dataset",
                            target="pbe-download-button",
                            placement="bottom",
                        ),
                        dbc.Button(
                            "r2SCAN",
                            href=f"{MATPES_SRC}/MatPES-R2SCAN-2025.2-charges.json?download=true",
                            class_name="me-1 download-button",
                            color="success",
                            external_link=True,
                            size="lg",
                            id="r2scan-download-button",
                        ),
                        dbc.Tooltip(
                            "Download r2SCAN dataset",
                            target="r2scan-download-button",
                            placement="bottom",
                        ),
                    ]
                ),
            ),
        ],
        fluid=True,
        className="py-3",
    ),
    className="p-3 bg-body-secondary rounded-3",
)


layout = dbc.Container(
    [
        jumbotron,
        dbc.Row(
            html.Div([dcc.Markdown(MARKDOWN_CONTENT, id="versions-table")]),
            className="mt-4",
        ),
    ]
)
