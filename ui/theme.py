"""Colors and the shared Plotly template for the dark theme.

Color is assigned by the job it does:
- sequential (one hue, dark -> light) for magnitude: implied vol, returns
- diverging (red <- grey -> blue) for sign: profit and loss
Event markers (spot, earnings) use neutral ink, not a data color.
"""

import plotly.graph_objects as go

SURFACE = "#1a1a19"
GRID = "#2e2e2b"
TEXT = "#f0efec"
TEXT_SECONDARY = "#c3c2b7"
TEXT_MUTED = "#8b8a83"
FILTERED_OUT = "#3a3a37"  # greyed-out heat map cells

SERIES_1 = "#3987e5"  # blue
SERIES_2 = "#d95926"  # orange

# Blue ramp, darkest (low) to lightest (high), so low values recede into the
# dark background and high values stand out.
_BLUE_RAMP = [
    "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
    "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb",
]
SEQUENTIAL = [[i / (len(_BLUE_RAMP) - 1), color] for i, color in enumerate(_BLUE_RAMP)]

# Losses red, gains blue, zero a neutral grey close to the background.
DIVERGING = [
    [0.00, "#f2a3a3"], [0.15, "#e66767"], [0.32, "#b53d3d"], [0.47, "#5a2e2c"],
    [0.50, "#383835"],
    [0.53, "#22406a"], [0.68, "#256abf"], [0.85, "#5598e7"], [1.00, "#b7d3f6"],
]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=TEXT_SECONDARY),
        title=dict(font=dict(size=15, color=TEXT), x=0, xanchor="left"),
        # automargin grows the margins to fit multi-line or wide tick labels,
        # so axis titles never sit on top of them.
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID,
                   automargin=True, title_standoff=10),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID,
                   automargin=True, title_standoff=10),
        hoverlabel=dict(bgcolor="#242422", bordercolor=GRID, font=dict(color=TEXT, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY)),
        margin=dict(l=60, r=20, t=50, b=50),
        colorway=[SERIES_1, SERIES_2],
    )
)

CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def colorbar(title: str, suffix: str = "", tickformat: str = "") -> dict:
    return dict(
        title=dict(text=title, side="right", font=dict(color=TEXT_SECONDARY)),
        ticksuffix=suffix,
        tickformat=tickformat,
        outlinewidth=0,
        thickness=14,
        tickfont=dict(color=TEXT_SECONDARY),
    )
