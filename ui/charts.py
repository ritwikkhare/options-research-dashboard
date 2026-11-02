"""Plotly figure builders. Each takes tidy DataFrames and returns a Figure."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analytics.screener import cell_grid
from analytics.surface import common_strikes, surface_grid
from ui import theme

IV_NOTES = {
    ("provider", "mid"): "",
    ("computed", "mid"): "IV recomputed from bid/ask mid",
    ("computed", "last"): "IV from last trade (market closed)",
}


def _base_figure(title: str, height: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=theme.PLOTLY_TEMPLATE, title=title, height=height)
    return fig


def _mark_spot(fig: go.Figure, spot: float, axis: str) -> None:
    """Dashed reference line at the current stock price."""
    line = dict(color=theme.TEXT_MUTED, width=1, dash="dash")
    label = f"Spot ${spot:,.2f}"
    if axis == "y":
        fig.add_hline(y=spot, line=line, annotation_text=label, annotation_position="top left",
                      annotation_font_color=theme.TEXT_SECONDARY)
    else:
        fig.add_vline(x=spot, line=line, annotation_text=label, annotation_position="top",
                      annotation_font_color=theme.TEXT_SECONDARY)


def _mark_earnings(fig: go.Figure, earnings_date: date | None, expiries) -> None:
    """Dotted vertical line at the next earnings date, if it falls in range."""
    if earnings_date is None or not len(expiries) or earnings_date > max(expiries):
        return
    x = pd.Timestamp(earnings_date)
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                  line=dict(color=theme.TEXT, width=1.5, dash="dot"))
    fig.add_annotation(x=x, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"Earnings {earnings_date:%b %d}", font=dict(color=theme.TEXT, size=12))


def _earnings_boundary(expiries: list[date], earnings_date: date | None) -> float | None:
    """Column edge between the last expiry before earnings and the first after.

    Every column right of this edge holds options that live through the
    earnings announcement, so their IV includes the expected earnings move.
    An expiry on the earnings date itself counts as before (US companies
    almost always report outside market hours).
    """
    if earnings_date is None:
        return None
    n_before = sum(expiry <= earnings_date for expiry in expiries)
    return None if n_before == len(expiries) else n_before - 0.5


def iv_heatmap(contracts: pd.DataFrame, spot: float, earnings_date: date | None, title: str) -> go.Figure:
    """Strikes (y) x expirations (x), colored by implied volatility.

    Both axes are evenly spaced (one row per strike, one column per expiry),
    like a broker's chain, rather than drawn to scale.
    """
    contracts = common_strikes(contracts)
    contracts = contracts.assign(
        note=[IV_NOTES.get(key, "") for key in zip(contracts["iv_source"], contracts["price_source"])],
        expiry_label=[f"{e:%b %d, %Y}" for e in contracts["expiry"]],
    )
    iv = surface_grid(contracts, "iv") * 100
    hover_fields = ["expiry_label", "dte", "bid", "ask", "volume", "open_interest", "note"]
    customdata = np.dstack(
        [surface_grid(contracts, field).reindex_like(iv).to_numpy(dtype=object) for field in hover_fields]
    )
    expiries, strikes = list(iv.columns), iv.index.to_numpy()
    columns, rows = np.arange(len(expiries)), np.arange(len(strikes))
    dte_by_expiry = contracts.groupby("expiry")["dte"].first()

    # Clip the color range to the 2nd-98th percentile so a few extreme wing
    # quotes don't wash out the rest of the surface.
    zmin, zmax = np.nanpercentile(iv.to_numpy(), [2, 98])

    fig = _base_figure(title, height=600)
    fig.add_trace(
        go.Heatmap(
            x=columns,
            y=rows,
            z=iv.to_numpy(),
            customdata=customdata,
            colorscale=theme.SEQUENTIAL,
            zmin=zmin,
            zmax=zmax,
            colorbar=theme.colorbar("Implied vol", "%"),
            hoverongaps=False,
            xgap=2,
            ygap=1 if len(strikes) < 80 else 0,
            text=np.broadcast_to(strikes[:, None], iv.shape),
            hovertemplate=(
                "<b>Strike $%{text:,.2f}</b> · %{customdata[0]} (%{customdata[1]} DTE)<br>"
                "IV <b>%{z:.1f}%</b><br>"
                "Bid $%{customdata[2]:.2f} / Ask $%{customdata[3]:.2f}<br>"
                "Volume %{customdata[4]:,.0f} · Open interest %{customdata[5]:,.0f}<br>"
                "<i>%{customdata[6]}</i><extra></extra>"
            ),
        )
    )

    recomputed = contracts[(contracts["iv_source"] == "computed") & (contracts["price_source"] == "mid")]
    if not recomputed.empty:
        column_of = {expiry: i for i, expiry in enumerate(expiries)}
        row_of = {strike: i for i, strike in enumerate(strikes)}
        fig.add_trace(
            go.Scatter(
                x=recomputed["expiry"].map(column_of),
                y=recomputed["strike"].map(row_of),
                mode="markers",
                marker=dict(symbol="circle-open", size=6, color=theme.TEXT, line=dict(width=1)),
                name="IV recomputed from bid/ask mid (provider IV missing or junk)",
                hoverinfo="skip",
            )
        )

    if strikes.min() <= spot <= strikes.max():
        fig.add_hline(
            y=float(np.interp(spot, strikes, rows)),
            line=dict(color=theme.TEXT, width=1, dash="dash"),
            annotation_text=f"Spot ${spot:,.2f}", annotation_position="top right",
            annotation_font_color=theme.TEXT,
        )

    boundary = _earnings_boundary(expiries, earnings_date)
    if boundary is not None:
        fig.add_vline(x=boundary, line=dict(color=theme.TEXT, width=2, dash="dot"))
        fig.add_annotation(x=boundary, y=1, yref="paper", yanchor="bottom", xanchor="left", showarrow=False,
                           text=f"  Earnings {earnings_date:%b %d} →", font=dict(color=theme.TEXT, size=12))

    tick_step = max(1, len(strikes) // 18)
    fig.update_layout(
        xaxis=dict(title="Expiration", tickvals=columns, showgrid=False,
                   ticktext=[f"{e:%b %d}<br>{dte_by_expiry[e]}d" for e in expiries]),
        yaxis=dict(title=dict(text="Strike", standoff=14), tickvals=rows[::tick_step], showgrid=False,
                   ticktext=[f"${s:,g}" for s in strikes[::tick_step]]),
        legend=dict(orientation="h", y=-0.16, x=0),
    )
    return fig


def screener_heatmap(rows: pd.DataFrame) -> go.Figure:
    """Ticker/strike (y) x expiration (x), colored by annualized return.

    Cells that fail a filter are drawn in flat grey underneath, with the
    reason in the hover, so near misses stay visible without competing with
    the trades that qualify.
    """
    rows = rows.assign(
        expiry_label=[f"{e:%b %d, %Y}" for e in rows["expiry"]],
        return_pct=rows["annualized_return"] * 100,
        otm_pct=rows["pct_otm"] * 100,
        iv_pct=rows["iv"] * 100,
    )
    returns = cell_grid(rows, "return_pct")
    passes = cell_grid(rows, "passes").reindex_like(returns).fillna(False).astype(bool).to_numpy()
    rows["label"] = [f"{t} ${k:,g} put" for t, k in zip(rows["ticker"], rows["strike"])]
    hover_fields = ["expiry_label", "dte", "premium", "collateral", "breakeven", "otm_pct", "delta",
                    "iv_pct", "bid", "ask", "open_interest", "fail_reason", "label"]
    customdata = np.dstack(
        [cell_grid(rows, field).reindex_like(returns).to_numpy(dtype=object) for field in hover_fields]
    )
    keys, expiries = list(returns.index), list(returns.columns)
    columns, row_ids = np.arange(len(expiries)), np.arange(len(keys))
    z = returns.to_numpy(dtype=float)
    exists = ~np.isnan(z)
    dte_by_expiry = rows.groupby("expiry")["dte"].first()

    fig = _base_figure("Annualized return on collateral", height=int(np.clip(120 + 26 * len(keys), 320, 1100)))
    fig.add_trace(
        go.Heatmap(
            x=columns, y=row_ids, z=np.where(exists & ~passes, 1.0, np.nan),
            colorscale=[[0, theme.FILTERED_OUT], [1, theme.FILTERED_OUT]], showscale=False,
            customdata=customdata, hoverongaps=False, xgap=2, ygap=2,
            hovertemplate=(
                "<b>%{customdata[12]}</b> · %{customdata[0]} (%{customdata[1]} DTE)<br>"
                "Filtered out: <i>%{customdata[11]}</i><extra></extra>"
            ),
        )
    )
    passing_z = np.where(passes, z, np.nan)
    # Print the return in each passing cell when the grid is small enough to read.
    cell_text = np.where(passes, np.char.mod("%.0f%%", np.nan_to_num(z)), "") if passes.sum() <= 150 else None
    fig.add_trace(
        go.Heatmap(
            x=columns, y=row_ids, z=passing_z,
            colorscale=theme.SEQUENTIAL,
            zmin=0, zmax=max(float(np.nanpercentile(passing_z, 98)), 1.0) if passes.any() else 1.0,
            colorbar=theme.colorbar("Annualized return", "%"),
            customdata=customdata, text=cell_text, texttemplate="%{text}" if cell_text is not None else None,
            hoverongaps=False, xgap=2, ygap=2, textfont=dict(size=11),
            hovertemplate=(
                "<b>%{customdata[12]}</b> · %{customdata[0]} (%{customdata[1]} DTE)<br>"
                "Annualized return <b>%{z:.1f}%</b><br>"
                "Premium $%{customdata[2]:,.0f} on $%{customdata[3]:,.0f} collateral<br>"
                "Breakeven $%{customdata[4]:,.2f} · %{customdata[5]:.1f}% below spot<br>"
                "Delta %{customdata[6]:.2f} · IV %{customdata[7]:.0f}%<br>"
                "Bid $%{customdata[8]:.2f} / Ask $%{customdata[9]:.2f} · OI %{customdata[10]:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    # Legend entry explaining grey cells (an invisible point carries the swatch).
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Fails a filter (hover for why)",
                             marker=dict(symbol="square", size=12, color=theme.FILTERED_OUT)))

    # Thin rules between tickers.
    tickers = [ticker for ticker, _ in keys]
    for i in range(1, len(tickers)):
        if tickers[i] != tickers[i - 1]:
            fig.add_hline(y=i - 0.5, line=dict(color=theme.TEXT_MUTED, width=1))

    fig.update_layout(
        xaxis=dict(tickvals=columns, showgrid=False, side="top",
                   ticktext=[f"{e:%b %d}<br>{dte_by_expiry[e]}d" for e in expiries]),
        yaxis=dict(tickvals=row_ids, ticktext=[f"{t}  ${s:,g}" for t, s in keys], showgrid=False,
                   autorange="reversed"),
        legend=dict(orientation="h", y=0, yanchor="top", x=0, yref="container"),
        margin=dict(t=90, b=40, l=110),
    )
    return fig


def smile_chart(smile: pd.DataFrame, spot: float, expiry: date) -> go.Figure:
    """IV by strike for one expiry; recomputed points drawn as open markers."""
    fig = _base_figure(f"IV smile · {expiry:%b %d, %Y}", height=360)
    symbols = np.where(smile["iv_source"] == "computed", "circle-open", "circle")
    fig.add_trace(
        go.Scatter(
            x=smile["strike"],
            y=smile["iv"] * 100,
            mode="lines+markers",
            line=dict(color=theme.SERIES_1, width=2),
            marker=dict(size=8, symbol=symbols, color=theme.SERIES_1),
            customdata=np.stack([smile["option_type"], smile["iv_source"]], axis=-1),
            hovertemplate="Strike $%{x:,.2f} %{customdata[0]}<br>IV %{y:.1f}% (%{customdata[1]})<extra></extra>",
        )
    )
    _mark_spot(fig, spot, axis="x")
    fig.update_layout(xaxis_title="Strike ($)", xaxis_tickprefix="$", yaxis_title="Implied vol",
                      yaxis_ticksuffix="%", showlegend=False)
    return fig


def term_structure_chart(term: pd.DataFrame, earnings_date: date | None) -> go.Figure:
    """ATM implied vol across expirations."""
    fig = _base_figure("ATM IV term structure", height=360)
    fig.add_trace(
        go.Scatter(
            x=[pd.Timestamp(e) for e in term["expiry"]],
            y=term["atm_iv"] * 100,
            mode="lines+markers",
            line=dict(color=theme.SERIES_1, width=2),
            marker=dict(size=8, color=theme.SERIES_1),
            customdata=term["dte"],
            hovertemplate="%{x|%b %d, %Y} (%{customdata} DTE)<br>ATM IV %{y:.1f}%<extra></extra>",
        )
    )
    _mark_earnings(fig, earnings_date, list(term["expiry"]))
    fig.update_layout(xaxis_title="Expiration", yaxis_title="ATM implied vol", yaxis_ticksuffix="%",
                      showlegend=False)
    return fig
