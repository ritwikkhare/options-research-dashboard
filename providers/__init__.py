"""Pick the market data source from configuration.

Set ``DATA_PROVIDER`` in ``.env`` (local) or Streamlit secrets (cloud):
    yfinance  free, ~15-minute delayed, no key needed (default)
    tradier   real-time with a Tradier account token (stub for now)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from providers.base import (
    CHAIN_COLUMNS,
    DataProvider,
    DataProviderError,
    NoOptionsError,
)

__all__ = [
    "CHAIN_COLUMNS",
    "DataProvider",
    "DataProviderError",
    "NoOptionsError",
    "get_provider",
]


def get_provider(name: str | None = None) -> DataProvider:
    load_dotenv()
    name = (name or os.getenv("DATA_PROVIDER") or "yfinance").strip().lower()

    if name == "yfinance":
        from providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider()

    if name == "tradier":
        from providers.tradier_provider import TradierProvider

        token = os.getenv("TRADIER_TOKEN", "").strip()
        if not token:
            raise DataProviderError("DATA_PROVIDER is 'tradier' but TRADIER_TOKEN is not set.")
        sandbox = os.getenv("TRADIER_SANDBOX", "true").strip().lower() == "true"
        return TradierProvider(token=token, sandbox=sandbox)

    raise DataProviderError(f"Unknown DATA_PROVIDER {name!r}. Use 'yfinance' or 'tradier'.")
