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

from analytics.screener import put_candidates
from analytics.surface import build_surface
from market_hours import days_to_expiry, now_et
from providers import DataProvider, DataProviderError, NoOptionsError, get_provider

QUOTE_TTL = 5 * 60          # seconds; matches the auto-refresh interval
CALENDAR_TTL = 60 * 60      # earnings dates, dividends and rates change slowly
FALLBACK_RISK_FREE_RATE = 0.04
# Modest parallelism: fast enough, without tripping Yahoo's rate limits.
TICKER_WORKERS = 3
CHAIN_WORKERS = 6


def provider() -> DataProvider:
    # Deliberately not cached: providers are cheap to create, and a cached
    # instance would outlive Streamlit's module reloads during development.
    return get_provider()


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


def fetch_snapshot(ticker: str, min_dte: int, max_dte: int) -> Snapshot:
    """Spot, chains in the DTE range, earnings and dividends for one ticker.

    Plain function (no Streamlit calls) so it can run in worker threads.
    Raises DataProviderError with a user-facing message when there is nothing
    to show (bad symbol, no listed options, nothing in the DTE range).
    """
    source = provider()
    spot, fetched_at = source.get_spot(ticker), now_et()
    expirations = source.get_expirations(ticker)
    if not expirations:
        raise NoOptionsError(f"{ticker} has no listed options.")

    in_range = [e for e in expirations if min_dte <= days_to_expiry(e) <= max_dte]
    if not in_range:
        raise NoOptionsError(
            f"No {ticker} expirations between {min_dte} and {max_dte} days out. "
            "Widen the expiration range in the sidebar."
        )

    def fetch_chain(expiry: date) -> pd.DataFrame | None:
        try:
            return source.get_chain(ticker, expiry)
        except DataProviderError:
            return None  # one bad expiry shouldn't blank the whole view

    with ThreadPoolExecutor(max_workers=CHAIN_WORKERS) as pool:
        chains = [c for c in pool.map(fetch_chain, in_range) if c is not None and not c.empty]
    if not chains:
        raise NoOptionsError(f"Couldn't load any {ticker} option quotes right now. Try refreshing.")

    return Snapshot(
        ticker=ticker,
        spot=spot,
        fetched_at=fetched_at,
        chain=pd.concat(chains, ignore_index=True),
        earnings_date=source.get_earnings_date(ticker),
        dividend_yield=source.get_dividend_yield(ticker, spot),
    )


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def load_snapshot(ticker: str, min_dte: int, max_dte: int) -> Snapshot:
    return fetch_snapshot(ticker, min_dte, max_dte)


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def load_watchlist(
    tickers: tuple[str, ...], min_dte: int, max_dte: int
) -> tuple[list[Snapshot], dict[str, str]]:
    """Snapshots for several tickers fetched in parallel.

    Returns the snapshots that loaded plus {ticker: friendly error} for the
    ones that didn't, so one bad symbol doesn't sink the whole watchlist.
    """

    def fetch(ticker: str) -> Snapshot | str:
        try:
            return fetch_snapshot(ticker, min_dte, max_dte)
        except DataProviderError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=TICKER_WORKERS) as pool:
        results = list(pool.map(fetch, tickers))
    snapshots = [r for r in results if isinstance(r, Snapshot)]
    errors = {t: r for t, r in zip(tickers, results) if isinstance(r, str)}
    return snapshots, errors


@st.cache_data(ttl=QUOTE_TTL, show_spinner="Solving implied volatilities...")
def cached_surface(
    chain: pd.DataFrame, spot: float, r: float, q: float, as_of: datetime, after_hours: bool
) -> pd.DataFrame:
    """build_surface is the slow step (one IV solve per contract), so cache it."""
    return build_surface(chain, spot, r, q, as_of, after_hours)


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def cached_put_candidates(
    chain: pd.DataFrame, ticker: str, spot: float, r: float, q: float, as_of: datetime, after_hours: bool
) -> pd.DataFrame:
    """Puts with IV, delta and return metrics, independent of filter settings."""
    return put_candidates(chain, ticker, spot, r, q, as_of, after_hours)
