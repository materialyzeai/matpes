"""Utility functions, e.g., pt heatmaps, etc."""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pymatgen.core.periodic_table import Element


@functools.lru_cache
def get_pt_df(include_artificial=False) -> pd.DataFrame:
    """
    Creates a DataFrame containing periodic table data.

    Returns:
        pd.DataFrame: DataFrame with element data including symbol, name, atomic number (Z),
                      electronegativity (X), group, period, category, and hover label.
    """
    with warnings.catch_warnings():
        # Suppress pymatgen warnings
        warnings.simplefilter("ignore")
        elements = [
            {
                "symbol": el.symbol,
                "name": el.long_name,
                "Z": el.Z,
                "X": el.X,
                "group": get_group(el),
                "period": get_period(el),
                "category": get_category(el),
            }
            for el in Element
            if (el.name not in ["D", "T"]) and (el.Z <= 92 or include_artificial)
        ]
    df = pd.DataFrame(elements)
    df["label"] = df.apply(lambda row: f"{row['Z']}<br>{row['symbol']}", axis=1)
    return df


def get_period(el: Element) -> int:
    """
    Determine the period for an element, with adjustments for rare earths.

    Args:
        el (Element): An element instance.

    Returns:
        int: The adjusted period number.
    """
    if el.is_actinoid or el.is_lanthanoid:
        return el.row + 2
    return el.row


def get_group(el: Element) -> int:
    """
    Determine the group for an element, with adjustments for rare earths.

    Args:
        el (Element): An element instance.

    Returns:
        int: The adjusted group number.
    """
    if el.is_actinoid:
        return el.group + el.Z - 89
    if el.is_lanthanoid:
        return el.group + el.Z - 57
    return el.group


def get_category(el: Element) -> str:
    """
    Categorize the element based on its type.

    Args:
        el (Element): An element instance.

    Returns:
        str: The category of the element (e.g., alkali, noble_gas, etc.).
    """
    if el.Z > 92:
        return "transuranic"
    for category in [
        "alkali",
        "alkaline",
        "actinoid",
        "lanthanoid",
        "halogen",
        "noble_gas",
        "metal",
        "chalcogen",
    ]:
        if getattr(el, f"is_{category}"):
            return category
    return "other"


def pt_heatmap(
    values: dict[str, float], label: str = "value", log: bool = False, include_artificial=False, colorscale="YlOrRd"
) -> go.Figure:
    """
    Generate a heatmap visualization of the periodic table.

    Args:
        values (dict[str, float]): Mapping of element symbols to values to visualize.
        label (str): Label for the values displayed.
        log (bool): Whether to use logarithmic scaling for the color axis.
        include_artificial (bool): Whether to include artificial elements. Defaults to False.
        colorscale (str): Colorscale to use for the heatmap. Defaults to "YlOrRd".

    Returns:
        plotly.graph_objects.Figure: A scatter plot representing the heatmap.
    """
    df = get_pt_df(include_artificial=include_artificial).copy()
    df[label] = df["symbol"].map(values) if values is not None else df["X"]
    if log:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            df[f"log10_{label}"] = np.log10(df[label])

    rows, cols = df["period"].to_numpy() - 1, df["group"].to_numpy() - 1

    # Initialize periodic table grid. Period 9 holds the actinoids.
    grid = np.full((9, 18), np.nan)
    missing = np.full((9, 18), np.nan)
    hover_texts = np.full((9, 18), "", dtype=object)

    grid[rows, cols] = df[f"log10_{label}" if log else label].to_numpy(dtype=float)
    records = list(zip(df["Z"], df["symbol"], df["name"], df[label], strict=True))
    hover_texts[rows, cols] = [f"{z}<br>{name}<br>{val}" for z, _, name, val in records]
    label_texts = [f"{z}<br>{sym}<br>{val}" for z, sym, _, val in records]
    is_missing = ~df["symbol"].isin(values.keys()) if values is not None else np.zeros(len(df), dtype=bool)
    missing[rows[is_missing], cols[is_missing]] = -1

    x, y = list(range(1, 19)), list(range(1, 10))

    # A single trace for all elements without values (shown in grey), a single trace for the heatmap and a single text
    # trace for labels. This is much faster to build and render than one trace / annotation per element.
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=missing, x=x, y=y, xgap=1, ygap=1, showscale=False, colorscale="Greys", hoverinfo="skip")
    )
    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=x,
            y=y,
            text=hover_texts,
            hoverinfo="text",
            showscale=True,
            colorscale="matter",
            xgap=1,
            ygap=1,
            coloraxis="coloraxis",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["group"],
            y=df["period"],
            text=label_texts,
            mode="text",
            textfont=dict(family="Arial", size=14, color="black", weight="bold"),
            hoverinfo="skip",
            showlegend=False,
            cliponaxis=False,
        )
    )
    # Hide x-axis
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0.5, 18.5])

    # Hide y-axis
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # Update layout
    fig.update_layout(
        title=None,
        xaxis=dict(title=None),  # Maintain 1:1 aspect ratio
        yaxis=dict(title=None, scaleanchor="x", scaleratio=1.33, range=[9.5, 0.5]),
        width=1200,
        height=900,
        colorscale={"sequential": colorscale},
        plot_bgcolor="white",
    )

    finite_logs = df[f"log10_{label}"][np.isfinite(df[f"log10_{label}"])] if log else None
    if log and len(finite_logs) > 0:
        max_log = int(finite_logs.max())
        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title=label,
                    tickmode="array",
                    tickvals=list(range(1, max_log + 1)),
                    ticktext=[f"1e{i}" for i in range(1, max_log + 1)],
                    tickfont=dict(size=14),
                )
            )
        )

    return fig
