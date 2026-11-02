"""Cached access to the data provider.

Every network call goes through ``st.cache_data`` with a 5-minute TTL, so
reruns, tab switches and multiple viewers share one fetch per ticker/expiry
instead of hammering the data source.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import streamlit as st

from analytics.surface import build_surface
from market_hours import days_to_expiry, now_et
from providers import DataProvider, DataProviderError, NoOptionsError, get_provider

QUOTE_TTL = 5 * 60          # seconds; matches the auto-refresh interval
CALENDAR_TTL = 60 * 60      # earnings dates, dividends and rates change slowly
FALLBACK_RISK_FREE_RATE = 0.04


def provider() -> DataProvider:
    # Deliberately not cached: providers are cheap to create, and a cached
    # instance would outlive Streamlit's module reloads during development.
    return get_provider()


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def load_spot(ticker: str) -> tuple[float, datetime]:
    """Spot price plus the time it was fetched (shown as 'Last updated')."""
    return provider().get_spot(ticker), now_et()


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def load_expirations(ticker: str) -> list[date]:
    return provider().get_expirations(ticker)


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def load_chains(ticker: str, expiries: tuple[date, ...]) -> pd.DataFrame:
    """Chains for several expiries, downloaded in parallel (one request each).

    Expiries that fail to load are skipped so one bad expiry doesn't blank
    the whole view; raises NoOptionsError only if every expiry fails.
    """
    source = provider()

    def fetch(expiry: date) -> pd.DataFrame | None:
        try:
            return source.get_chain(ticker, expiry)
        except DataProviderError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        chains = [chain for chain in pool.map(fetch, expiries) if chain is not None and not chain.empty]
    if not chains:
        raise NoOptionsError(f"Couldn't load any {ticker} option quotes right now. Try refreshing.")
    return pd.concat(chains, ignore_index=True)


@st.cache_data(ttl=CALENDAR_TTL, show_spinner=False)
def load_earnings_date(ticker: str) -> date | None:
    return provider().get_earnings_date(ticker)


@st.cache_data(ttl=CALENDAR_TTL, show_spinner=False)
def load_dividend_yield(ticker: str) -> float:
    spot, _ = load_spot(ticker)
    return provider().get_dividend_yield(ticker, spot)


@st.cache_data(ttl=CALENDAR_TTL, show_spinner=False)
def load_risk_free_rate() -> float:
    return provider().get_risk_free_rate() or FALLBACK_RISK_FREE_RATE


@dataclass
class Snapshot:
    """Everything the views need about one ticker at one point in time."""

    ticker: str
    spot: float
    fetched_at: datetime
    chain: pd.DataFrame
    earnings_date: date | None
    dividend_yield: float


def load_snapshot(ticker: str, min_dte: int, max_dte: int) -> Snapshot:
    """Spot, chains in the DTE range, earnings and dividends for one ticker.

    Raises DataProviderError with a user-facing message when there is nothing
    to show (bad symbol, no listed options, nothing in the DTE range).
    """
    spot, fetched_at = load_spot(ticker)
    expirations = load_expirations(ticker)
    if not expirations:
        raise NoOptionsError(f"{ticker} has no listed options.")

    in_range = [e for e in expirations if min_dte <= days_to_expiry(e) <= max_dte]
    if not in_range:
        raise NoOptionsError(
            f"No {ticker} expirations between {min_dte} and {max_dte} days out. "
            "Widen the expiration range in the sidebar."
        )

    return Snapshot(
        ticker=ticker,
        spot=spot,
        fetched_at=fetched_at,
        chain=load_chains(ticker, tuple(in_range)),
        earnings_date=load_earnings_date(ticker),
        dividend_yield=load_dividend_yield(ticker),
    )


@st.cache_data(ttl=QUOTE_TTL, show_spinner="Solving implied volatilities...")
def cached_surface(
    chain: pd.DataFrame, spot: float, r: float, q: float, as_of: datetime, after_hours: bool
) -> pd.DataFrame:
    """build_surface is the slow step (one IV solve per contract), so cache it."""
    return build_surface(chain, spot, r, q, as_of, after_hours)
