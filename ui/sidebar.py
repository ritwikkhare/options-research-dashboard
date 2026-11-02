"""Global controls shared by every tab."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import streamlit as st

from market_hours import is_market_open
from ui.data import QUOTE_TTL, load_risk_free_rate

PRESET_TICKERS = ["NVDA", "PLTR", "AAPL"]
DEFAULT_DTE_RANGE = (7, 60)
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


@dataclass
class Settings:
    ticker: str
    min_dte: int
    max_dte: int
    risk_free_rate: float  # decimal
    market_open: bool


def _choose_preset(ticker: str) -> None:
    st.session_state.ticker = ticker
    st.session_state.ticker_input = ""


def _choose_typed_ticker() -> None:
    typed = st.session_state.ticker_input.strip().upper()
    if TICKER_PATTERN.match(typed):
        st.session_state.ticker = typed
        st.session_state.ticker_error = None
    elif typed:
        st.session_state.ticker_error = f"'{typed}' doesn't look like a ticker symbol."


def render_sidebar() -> tuple[Settings, st.delta_generator.DeltaGenerator]:
    """Draw the sidebar. Returns the settings and a slot for 'Last updated'."""
    st.session_state.setdefault("ticker", PRESET_TICKERS[0])
    st.session_state.setdefault("ticker_error", None)
    sidebar = st.sidebar

    sidebar.header("Ticker")
    for column, ticker in zip(sidebar.columns(len(PRESET_TICKERS)), PRESET_TICKERS):
        column.button(
            ticker,
            on_click=_choose_preset,
            args=(ticker,),
            type="primary" if st.session_state.ticker == ticker else "secondary",
            width="stretch",
        )
    sidebar.text_input(
        "Any other ticker",
        key="ticker_input",
        placeholder="e.g. MSFT, then Enter",
        on_change=_choose_typed_ticker,
    )
    if st.session_state.ticker_error:
        sidebar.caption(f":orange[{st.session_state.ticker_error}]")

    sidebar.header("Market inputs")
    min_dte, max_dte = sidebar.slider(
        "Days to expiration",
        min_value=0,
        max_value=365,
        value=DEFAULT_DTE_RANGE,
        help="Only expirations inside this range are loaded and plotted.",
    )
    default_rate_pct = round(load_risk_free_rate() * 100, 2)
    rate_pct = sidebar.number_input(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=default_rate_pct,
        step=0.05,
        format="%.2f",
        help="Defaults to the current 13-week Treasury bill yield (^IRX). Used for "
        "Black-Scholes pricing, Greeks and implied vol.",
    )

    sidebar.header("Data")
    market_open = is_market_open()
    last_updated = sidebar.empty()
    sidebar.button("Refresh now", on_click=st.cache_data.clear, width="stretch")
    sidebar.caption(
        f"Auto-refreshes every {QUOTE_TTL // 60} minutes while the market is open."
        if market_open
        else "Market closed: auto-refresh paused."
    )
    sidebar.caption("Research only. Quotes are delayed ~15 minutes; no orders are ever placed.")

    settings = Settings(
        ticker=st.session_state.ticker,
        min_dte=min_dte,
        max_dte=max_dte,
        risk_free_rate=rate_pct / 100,
        market_open=market_open,
    )
    return settings, last_updated


def show_last_updated(slot, fetched_at: datetime | None) -> None:
    if fetched_at is None:
        slot.caption("Last updated: n/a")
    else:
        slot.caption(f"Last updated: **{fetched_at:%b %d, %I:%M:%S %p} ET**")
