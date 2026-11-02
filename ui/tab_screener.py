"""Tab 2: cash-secured put screener across a watchlist."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from analytics.screener import ScreenerFilters, apply_filters, heatmap_rows, ticker_summary, top_candidates
from ui import charts
from ui.data import cached_put_candidates, load_watchlist
from ui.sidebar import Settings
from ui.theme import CHART_CONFIG

# The big three plus cheaper names, so something fits under a $2,000 cap.
DEFAULT_WATCHLIST = "NVDA, PLTR, AAPL, SOFI, F, T, INTC, RIVN"
MAX_TICKERS = 15
DEFAULTS = ScreenerFilters()


def parse_watchlist(text: str) -> tuple[list[str], list[str]]:
    """Split user input into valid tickers (deduplicated, in order) and rejects."""
    tokens = [t.strip().upper() for t in re.split(r"[,\s]+", text) if t.strip()]
    valid = [t for t in dict.fromkeys(tokens) if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", t)]
    rejected = [t for t in tokens if t not in valid]
    return valid[:MAX_TICKERS], rejected


def _filter_controls() -> tuple[list[str], ScreenerFilters]:
    watch_col, collateral_col, oi_col = st.columns([3, 1, 1])
    tickers, rejected = parse_watchlist(
        watch_col.text_input("Watchlist (comma-separated)", value=DEFAULT_WATCHLIST, key="screener_watchlist")
    )
    max_collateral = collateral_col.number_input(
        "Max collateral ($)", min_value=100, max_value=1_000_000, value=int(DEFAULTS.max_collateral), step=500,
        help="Cash the broker sets aside: strike x 100 per contract.",
    )
    min_open_interest = oi_col.number_input(
        "Min open interest", min_value=0, max_value=100_000, value=DEFAULTS.min_open_interest, step=50,
        help="Contracts already open. Low open interest usually means a wide, hard-to-fill market.",
    )

    delta_col, spread_col, earnings_col = st.columns([3, 1, 1])
    min_delta, max_delta = delta_col.slider(
        "|Delta| range", 0.0, 1.0, (DEFAULTS.min_abs_delta, DEFAULTS.max_abs_delta), step=0.01,
        help="Absolute put delta, a rough probability of the put finishing in the money "
        "(i.e. of being assigned the shares).",
    )
    max_spread_pct = spread_col.number_input(
        "Max spread (% of mid)", min_value=1, max_value=100, value=int(DEFAULTS.max_spread_pct * 100), step=1,
    )
    include_earnings = earnings_col.toggle(
        "Include earnings", value=False,
        help="Off: puts that expire after the next earnings date are greyed out, because "
        "earnings can gap the stock through your strike.",
    )
    if rejected:
        st.caption(f":orange[Ignored: {', '.join(rejected)} (not valid ticker symbols).]")

    return tickers, ScreenerFilters(
        max_collateral=float(max_collateral),
        min_abs_delta=min_delta,
        max_abs_delta=max_delta,
        min_open_interest=int(min_open_interest),
        max_spread_pct=max_spread_pct / 100,
        include_earnings=include_earnings,
    )


def _candidates_table(top: pd.DataFrame, show_price_source: bool) -> None:
    table = pd.DataFrame(
        {
            "Ticker": top["ticker"],
            "Strike": top["strike"],
            "Expiry": top["expiry"],
            "DTE": top["dte"],
            "Annualized": top["annualized_return"] * 100,
            "Premium": top["premium"],
            "Collateral": top["collateral"],
            "Breakeven": top["breakeven"],
            "% OTM": top["pct_otm"] * 100,
            "Delta": top["delta"],
            "IV": top["iv"] * 100,
            "Open int.": top["open_interest"],
            "Spread": top["spread_pct"] * 100,
        }
    )
    if show_price_source:
        table["Priced from"] = top["price_source"].map({"mid": "bid/ask mid", "last": "last trade"})
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Strike": st.column_config.NumberColumn(format="$%.2f"),
            "Expiry": st.column_config.DateColumn(format="MMM DD, YYYY"),
            "Annualized": st.column_config.NumberColumn(format="%.1f%%", help="(premium / collateral) x 365 / DTE"),
            "Premium": st.column_config.NumberColumn(format="$%.0f", help="Mid price x 100, per contract"),
            "Collateral": st.column_config.NumberColumn(format="dollar"),
            "Breakeven": st.column_config.NumberColumn(format="$%.2f", help="Strike minus premium per share"),
            "% OTM": st.column_config.NumberColumn(format="%.1f%%", help="How far the strike is below spot"),
            "Delta": st.column_config.NumberColumn(format="%.2f"),
            "IV": st.column_config.NumberColumn(format="%.0f%%"),
            "Open int.": st.column_config.NumberColumn(format="localized"),
            "Spread": st.column_config.NumberColumn(format="%.1f%%", help="Bid/ask spread as % of mid"),
        },
    )


def render(settings: Settings) -> None:
    tickers, filters = _filter_controls()
    if not tickers:
        st.info("Add at least one ticker to the watchlist.")
        return

    with st.spinner(f"Loading option chains for {len(tickers)} tickers..."):
        snapshots, errors = load_watchlist(tuple(tickers), settings.min_dte, settings.max_dte)
    if not snapshots:
        st.info("None of the watchlist tickers returned option data. " + " ".join(errors.values()))
        return

    after_hours = not settings.market_open
    frames = [
        cached_put_candidates(s.chain, s.ticker, s.spot, settings.risk_free_rate, s.dividend_yield,
                              s.fetched_at, after_hours)
        for s in snapshots
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        st.info("No puts with a usable price in this expiration range. Check back during market hours.")
        return
    candidates = pd.concat(frames, ignore_index=True)

    screened = apply_filters(candidates, filters, {s.ticker: s.earnings_date for s in snapshots})
    top = top_candidates(screened, n=20)
    summary = ticker_summary(screened, filters)

    passing_col, best_col, fit_col = st.columns(3)
    passing_col.metric("Puts passing all filters", f"{int(screened['passes'].sum()):,}")
    best_col.metric("Best annualized return", f"{top['annualized_return'].iloc[0]:.1%}" if not top.empty else "n/a")
    fit_col.metric("Tickers with a fit", f"{int((summary['passing'] > 0).sum())} of {len(summary)}")

    if after_hours:
        st.caption(
            ":orange[Market closed:] premiums use each put's last trade price instead of a live "
            "bid/ask mid, and the spread filter is skipped. Treat returns as indicative until the open."
        )

    rows = heatmap_rows(screened)
    if rows.empty:
        st.info(
            "No put passes every filter. Try a wider delta range, a higher collateral cap, "
            "or including earnings. The watchlist summary below shows what blocks each ticker."
        )
    else:
        st.plotly_chart(charts.screener_heatmap(rows), config=CHART_CONFIG, theme=None, width="stretch")

    st.subheader("Top 20 candidates")
    if top.empty:
        st.caption("Nothing passes every filter yet.")
    else:
        _candidates_table(top, show_price_source=after_hours)

    with st.expander("Watchlist summary", expanded=bool(errors)):
        for ticker, message in errors.items():
            st.caption(f"**{ticker}**: {message}")
        st.dataframe(
            summary.rename(columns={"ticker": "Ticker", "spot": "Spot", "passing": "Passing", "note": "Note"}),
            hide_index=True,
            width="stretch",
            column_config={"Spot": st.column_config.NumberColumn(format="$%.2f")},
        )

    with st.expander("How the screener works"):
        st.markdown(
            r"""
Selling a **cash-secured put** pays you a premium now; in return you agree to buy 100 shares at the
strike if the stock closes below it at expiry. Your broker holds **strike x 100** in cash meanwhile.

- **Annualized return** $= \dfrac{\text{mid} \times 100}{\text{strike} \times 100} \times \dfrac{365}{\text{DTE}}$,
  so short and long expiries compare fairly. It assumes you could repeat the trade at the same rate,
  which you usually can't. Use it to rank, not as a forecast.
- **Breakeven** = strike - premium per share. Below it, the position loses money at expiry.
- **Delta** works as a rough probability of assignment. 0.15-0.35 balances premium against the odds
  of ending up owning the stock.
- **Earnings:** an earnings gap can blow straight through a strike, so puts that expire after
  the next report are greyed out unless you include them.
- Premiums use the **mid**; a real fill is usually a bit worse, especially when the spread is wide.
            """
        )
