"""Tab 3: portfolio Greeks, collateral, assignment flags and a P&L scenario grid.

Positions live only in this browser session. Nothing is sent anywhere, and
there is no brokerage connection.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.risk import (
    POSITION_COLUMNS,
    empty_positions,
    lookup_iv,
    parse_positions,
    pnl_grid,
    portfolio_totals,
    value_positions,
)
from analytics.surface import select_side
from market_hours import days_to_expiry, now_et
from ui import charts
from ui.data import cached_surface, load_watchlist
from ui.sidebar import Settings
from ui.theme import CHART_CONFIG

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_positions.csv"
DEFAULT_COLLATERAL_CAP = 2_000
FALLBACK_IV = 0.50  # only when a ticker has no usable option quotes at all


def _money(x: float, decimals: int = 2) -> str:
    """-$26.07 rather than $-26.07."""
    return f"{'-' if x < 0 else ''}${abs(x):,.{decimals}f}"


def _md(text: str) -> str:
    """Escape dollar signs: Streamlit markdown treats a pair of $ as LaTeX math."""
    return text.replace("$", r"\$")


def _load_sample() -> pd.DataFrame:
    positions, _ = parse_positions(pd.read_csv(SAMPLE_CSV), now_et().date())
    return positions


def _replace_positions(positions: pd.DataFrame) -> None:
    """Swap the editor's contents (bumping its key resets its internal edits)."""
    st.session_state.positions = positions
    st.session_state.positions_version = st.session_state.get("positions_version", 0) + 1


def _positions_editor() -> pd.DataFrame:
    """Upload / sample / clear controls, the editable table, and the add form."""
    if "positions" not in st.session_state:
        _replace_positions(_load_sample())

    upload_col, sample_col, clear_col, template_col = st.columns([3, 1, 1, 1], vertical_alignment="bottom")
    upload = upload_col.file_uploader("Upload positions CSV", type="csv", key="positions_upload")
    if upload is not None and st.session_state.get("uploaded_name") != upload.name:
        st.session_state.uploaded_name = upload.name
        try:
            uploaded, problems = parse_positions(pd.read_csv(upload), now_et().date())
        except Exception:
            uploaded, problems = empty_positions(), ["Couldn't read that file as a CSV."]
        st.session_state.upload_problems = problems
        _replace_positions(uploaded)
    if sample_col.button("Load sample", width="stretch"):
        st.session_state.upload_problems = []
        _replace_positions(_load_sample())
    if clear_col.button("Clear all", width="stretch"):
        st.session_state.upload_problems = []
        _replace_positions(empty_positions())
    template_col.download_button(
        "CSV template", SAMPLE_CSV.read_bytes(), file_name="positions.csv", mime="text/csv", width="stretch",
    )
    for problem in st.session_state.get("upload_problems", []):
        st.warning(_md(f"Skipped from upload. {problem}"))

    edited = st.data_editor(
        st.session_state.positions[POSITION_COLUMNS],
        key=f"positions_editor_{st.session_state.positions_version}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", required=True),
            "type": st.column_config.SelectboxColumn("Type", options=["call", "put", "stock"], required=True),
            "side": st.column_config.SelectboxColumn("Side", options=["long", "short"], required=True),
            "strike": st.column_config.NumberColumn("Strike", format="$%.2f", min_value=0),
            "expiry": st.column_config.DateColumn("Expiry", format="MMM DD, YYYY"),
            "quantity": st.column_config.NumberColumn(
                "Quantity", min_value=0, help="Contracts for options, shares for stock"),
            "entry_price": st.column_config.NumberColumn(
                "Entry price", format="$%.2f", min_value=0, help="Per share: the option premium or stock price"),
        },
    )

    with st.expander("Add a position"):
        with st.form("add_position", clear_on_submit=True, border=False):
            c1, c2, c3, c4 = st.columns(4)
            ticker = c1.text_input("Ticker")
            kind = c2.selectbox("Type", ["put", "call", "stock"])
            side = c3.selectbox("Side", ["short", "long"])
            quantity = c4.number_input("Quantity", min_value=1, value=1, step=1)
            c5, c6, c7, _ = st.columns(4)
            strike = c5.number_input("Strike (options)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
            expiry = c6.date_input("Expiry (options)", value=None)
            entry_price = c7.number_input("Entry price", min_value=0.0, value=0.0, step=0.05, format="%.2f")
            if st.form_submit_button("Add position"):
                new = {
                    "ticker": ticker, "type": kind, "side": side, "quantity": quantity, "entry_price": entry_price,
                    "strike": strike if kind != "stock" else math.nan,
                    "expiry": expiry if kind != "stock" else None,
                }
                parsed, problems = parse_positions(pd.DataFrame([new]), now_et().date())
                if problems:
                    st.warning(problems[0].replace("Row 1: ", "Not added: "))
                else:
                    _replace_positions(pd.concat([edited, parsed], ignore_index=True))
                    st.rerun()
    return edited


def _greeks_table(valued: pd.DataFrame, totals: dict) -> None:
    table = pd.DataFrame(
        {
            "Position": valued["label"],
            # Text, not numbers, so stock rows read blank instead of "None".
            "DTE": [f"{int(d)}" if t != "stock" else "" for d, t in zip(valued["dte"], valued["type"])],
            "IV": [f"{iv:.0%}" if t != "stock" else "" for iv, t in zip(valued["iv"], valued["type"])],
            "Value": valued["market_value"],
            "P&L": valued["pnl"],
            "Delta (sh)": valued["delta"],
            "Gamma": valued["gamma"],
            "Theta $/day": valued["theta"],
            "Vega $/pt": valued["vega"],
            "Collateral": valued["collateral"],
            "Max loss": valued["max_loss"].map(lambda x: "Unlimited" if math.isinf(x) else f"${x:,.0f}"),
            "Flag": valued["flag"].map(lambda f: f"⚠ {f}" if f else ""),
        }
    )
    total_row = {
        "Position": "Total", "DTE": "", "IV": "", "Value": totals["market_value"], "P&L": totals["pnl"],
        "Delta (sh)": totals["delta"], "Gamma": totals["gamma"], "Theta $/day": totals["theta"],
        "Vega $/pt": totals["vega"], "Collateral": totals["collateral"],
        "Max loss": "Unlimited" if math.isinf(totals["max_loss"]) else f"${totals['max_loss']:,.0f}",
        "Flag": "",
    }
    table = pd.concat([table, pd.DataFrame([total_row])], ignore_index=True)
    money = st.column_config.NumberColumn(format="dollar")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Value": money,
            "P&L": money,
            "Delta (sh)": st.column_config.NumberColumn(
                format="%.1f", help="Share-equivalent exposure: P&L change for a $1 move in the stock"),
            "Gamma": st.column_config.NumberColumn(format="%.2f", help="Change in delta for a $1 move"),
            "Theta $/day": st.column_config.NumberColumn(format="dollar", help="P&L from one day passing"),
            "Vega $/pt": st.column_config.NumberColumn(format="dollar", help="P&L for a 1-point rise in IV"),
            "Collateral": st.column_config.NumberColumn(format="dollar"),
        },
    )


def render(settings: Settings) -> None:
    edited = _positions_editor()
    now = now_et()
    positions, problems = parse_positions(edited, now.date())
    for problem in problems:
        st.warning(_md(f"Skipped. {problem}"))
    if positions.empty:
        st.info("Add positions above (or load the sample) to see portfolio risk.")
        return

    cap_col, shift_col = st.columns([1, 2])
    collateral_cap = cap_col.number_input("Collateral cap ($)", min_value=0, value=DEFAULT_COLLATERAL_CAP, step=500)
    iv_shift_pts = shift_col.slider(
        "IV shift (vol points)", -20, 20, 0, step=1,
        help="Shift every option's implied vol up or down in parallel, e.g. +10 to see an IV spike.",
    )

    # Market data: one snapshot per ticker, covering every position's expiry.
    tickers = tuple(positions["ticker"].unique())
    option_expiries = [e for e in positions["expiry"] if e is not None]
    max_dte = max([days_to_expiry(e, now) for e in option_expiries], default=0) + 7
    with st.spinner("Loading market data for your positions..."):
        snapshots, errors = load_watchlist(tickers, 0, max_dte)
    snapshot_by_ticker = {s.ticker: s for s in snapshots}

    # Tickers that failed to load are skipped with a warning; the rest still render.
    missing = [t for t in tickers if t not in snapshot_by_ticker]
    if missing:
        for ticker in missing:
            st.warning(_md(f"{ticker} positions skipped: {errors.get(ticker, 'no market data')}"))
        positions = positions[positions["ticker"].isin(snapshot_by_ticker)]
        if positions.empty:
            return

    after_hours = not settings.market_open
    ivs, fallback_used = [], set()
    for row in positions.itertuples(index=False):
        if row.type == "stock":
            ivs.append(0.0)
            continue
        s = snapshot_by_ticker[row.ticker]
        contracts = select_side(
            cached_surface(s.chain, s.spot, settings.risk_free_rate, s.dividend_yield, s.fetched_at, after_hours),
            "otm",
        )
        iv = lookup_iv(contracts, row.strike, row.expiry)
        if math.isnan(iv):
            iv = FALLBACK_IV
            fallback_used.add(row.ticker)
        ivs.append(iv)
    if fallback_used:
        st.caption(f":orange[No usable option quotes for {', '.join(sorted(fallback_used))}; "
                   f"assumed {FALLBACK_IV:.0%} IV.]")

    valued = value_positions(
        positions,
        {t: s.spot for t, s in snapshot_by_ticker.items()},
        {t: s.dividend_yield for t, s in snapshot_by_ticker.items()},
        ivs, settings.risk_free_rate, now,
    )
    totals = portfolio_totals(valued)

    delta_col, theta_col, vega_col, collateral_col, loss_col = st.columns(5)
    delta_col.metric("Net delta", f"{totals['delta']:,.0f} sh", help="Share-equivalent exposure across all positions")
    theta_col.metric("Theta", f"{_money(totals['theta'])}/day", help="P&L from one day passing, all else equal")
    vega_col.metric("Vega", f"{_money(totals['vega'])}/pt", help="P&L if implied vol rises 1 point")
    headroom = collateral_cap - totals["collateral"]
    collateral_col.metric(
        "Collateral used", f"${totals['collateral']:,.0f}",
        delta=_md(f"${headroom:,.0f} left of ${collateral_cap:,.0f}" if headroom >= 0
                  else f"${-headroom:,.0f} over ${collateral_cap:,.0f} cap"),
        delta_color="off", delta_arrow="off",
    )
    loss_col.metric(
        "Max loss", "Unlimited" if math.isinf(totals["max_loss"]) else f"${totals['max_loss']:,.0f}",
        help="Sum of each position's worst case at expiry. Conservative for spreads, which offset.",
    )
    if collateral_cap:
        st.progress(min(totals["collateral"] / collateral_cap, 1.0),
                    text=f"Collateral: {totals['collateral'] / collateral_cap:.0%} of cap")
    if totals["collateral"] > collateral_cap:
        st.error(_md(f":material/error: Over the collateral cap by ${totals['collateral'] - collateral_cap:,.0f}."))

    for row in valued[valued["flag"] != ""].itertuples(index=False):
        icon = ":material/error:" if ("likely" in row.flag or "Uncovered" in row.flag) else ":material/warning:"
        st.warning(_md(f"{icon} **{row.label}**: {row.flag} (spot ${row.spot:,.2f})"))

    _greeks_table(valued, totals)

    grid = pnl_grid(valued, settings.risk_free_rate, now, iv_shift=iv_shift_pts / 100)
    st.plotly_chart(charts.pnl_heatmap(grid, option_expiries, iv_shift_pts),
                    config=CHART_CONFIG, theme=None, width="stretch")

    with st.expander("How to read this view"):
        st.markdown(
            """
- **Greeks** are for the whole position (per-share Greeks x 100 per contract x quantity, negative
  for short positions). Net delta of 50 behaves like owning 50 shares. Positive theta means time
  decay is paying you; negative vega means you lose if implied vol rises.
- **Collateral** counts cash-secured puts at strike x 100. Covered calls need none (the shares are
  the collateral). Spreads and naked calls are not margin-modelled.
- **Flags:** a short put that is in the money, or within 5% of its strike with under 7 days left,
  is at real risk of assignment: you'd buy 100 shares per contract at the strike.
- **P&L grid:** each cell reprices every position with Black-Scholes for that price move and date,
  holding implied vol constant (plus the IV shift). Every ticker moves by the same %; after an
  option's expiry it is held at its intrinsic value. Grey is break-even, blue profit, red loss.
            """
        )
