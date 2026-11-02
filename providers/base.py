"""The data-provider interface every market data source implements.

The UI and analytics only ever talk to ``DataProvider``, so switching from
yfinance (free, delayed) to a real-time source is a one-line config change.
Providers are read-only by design: there are no order or account methods.

Every provider returns option chains as a DataFrame with ``CHAIN_COLUMNS``:

    contract       provider's contract symbol, e.g. "NVDA261120P00150000"
    option_type    "call" or "put"
    expiry         datetime.date of expiration
    strike         float
    bid, ask, last float, per share (NaN if missing)
    volume         contracts traded today (0 if none)
    open_interest  open contracts as of the prior close (0 if none)
    provider_iv    the source's implied vol as a decimal (NaN if missing)
    last_trade     UTC timestamp of the last trade (NaT if unknown)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

CHAIN_COLUMNS = [
    "contract",
    "option_type",
    "expiry",
    "strike",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "provider_iv",
    "last_trade",
]


class DataProviderError(Exception):
    """A data problem with a message that is safe to show to the user."""


class NoOptionsError(DataProviderError):
    """The ticker exists but has no listed options (or none in range)."""


class DataProvider(ABC):
    """Read-only source of quotes, option chains and calendar data."""

    name: str = "base"
    is_realtime: bool = False

    @abstractmethod
    def get_spot(self, ticker: str) -> float:
        """Latest price of the underlying. Raises DataProviderError."""

    @abstractmethod
    def get_expirations(self, ticker: str) -> list[date]:
        """All listed expiration dates, sorted. Empty list if no options."""

    @abstractmethod
    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        """Calls and puts for one expiry, with CHAIN_COLUMNS."""

    def get_earnings_date(self, ticker: str) -> date | None:
        """Next scheduled earnings date, if the source knows it."""
        return None

    def get_dividend_yield(self, ticker: str, spot: float) -> float:
        """Continuous dividend yield used in pricing (decimal). Defaults to 0."""
        return 0.0

    def get_risk_free_rate(self) -> float | None:
        """Short-term Treasury yield (decimal), or None if unavailable."""
        return None

    def get_chains(self, ticker: str, expiries: list[date]) -> pd.DataFrame:
        """Chains for several expiries combined; skips expiries that fail to load."""
        frames = []
        for expiry in expiries:
            try:
                chain = self.get_chain(ticker, expiry)
            except DataProviderError:
                continue
            if not chain.empty:
                frames.append(chain)
        if not frames:
            raise NoOptionsError(f"No option quotes could be loaded for {ticker}.")
        return pd.concat(frames, ignore_index=True)[CHAIN_COLUMNS]


def empty_chain() -> pd.DataFrame:
    return pd.DataFrame(columns=CHAIN_COLUMNS)
