"""yfinance data provider: free, no API key, quotes delayed ~15 minutes.

Quirks this class smooths over:
- Bid/ask come back as 0.0 outside market hours; we pass them through and
  let the analytics layer decide how to handle stale quotes.
- Implied vol is often junk (e.g. 0.00001) on illiquid contracts.
- Unknown tickers raise assorted exceptions or return empty results, so every
  call is wrapped and converted to a DataProviderError with a friendly message.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from market_hours import now_et
from providers.base import CHAIN_COLUMNS, DataProvider, DataProviderError, empty_chain

# yfinance logs its own HTTP errors (e.g. 404 for unknown tickers); we report
# problems to the user ourselves.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

TREASURY_BILL_SYMBOL = "^IRX"  # 13-week T-bill yield, quoted in percent


def normalize_yf_chain(raw: pd.DataFrame, option_type: str, expiry: date) -> pd.DataFrame:
    """Map a yfinance calls/puts table onto the shared CHAIN_COLUMNS schema."""
    if raw is None or raw.empty:
        return empty_chain()

    chain = pd.DataFrame(
        {
            "contract": raw["contractSymbol"],
            "option_type": option_type,
            "expiry": expiry,
            "strike": pd.to_numeric(raw["strike"], errors="coerce"),
            "bid": pd.to_numeric(raw["bid"], errors="coerce"),
            "ask": pd.to_numeric(raw["ask"], errors="coerce"),
            "last": pd.to_numeric(raw["lastPrice"], errors="coerce"),
            "volume": pd.to_numeric(raw["volume"], errors="coerce").fillna(0),
            "open_interest": pd.to_numeric(raw["openInterest"], errors="coerce").fillna(0),
            "provider_iv": pd.to_numeric(raw["impliedVolatility"], errors="coerce"),
            "last_trade": pd.to_datetime(raw["lastTradeDate"], utc=True, errors="coerce"),
        }
    )
    return chain.dropna(subset=["strike"])[CHAIN_COLUMNS].reset_index(drop=True)


class YFinanceProvider(DataProvider):
    name = "yfinance"
    is_realtime = False

    def get_spot(self, ticker: str) -> float:
        try:
            price = yf.Ticker(ticker).fast_info["last_price"]
        except Exception as exc:
            raise DataProviderError(f"Couldn't find a price for '{ticker}'. Check the symbol.") from exc
        if price is None or not math.isfinite(price) or price <= 0:
            raise DataProviderError(f"Couldn't find a price for '{ticker}'. Check the symbol.")
        return float(price)

    def get_expirations(self, ticker: str) -> list[date]:
        try:
            raw = yf.Ticker(ticker).options
        except Exception as exc:
            raise DataProviderError(f"Couldn't load option expirations for '{ticker}'.") from exc
        return sorted(date.fromisoformat(day) for day in raw)

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        try:
            chain = yf.Ticker(ticker).option_chain(expiry.isoformat())
        except Exception as exc:
            raise DataProviderError(f"Couldn't load the {expiry:%b %d} chain for '{ticker}'.") from exc
        frames = [
            normalize_yf_chain(chain.calls, "call", expiry),
            normalize_yf_chain(chain.puts, "put", expiry),
        ]
        frames = [frame for frame in frames if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else empty_chain()

    def get_earnings_date(self, ticker: str) -> date | None:
        try:
            calendar = yf.Ticker(ticker).calendar
        except Exception:
            return None
        if not isinstance(calendar, dict):
            return None
        today = now_et().date()
        upcoming = [day for day in calendar.get("Earnings Date") or [] if day >= today]
        return min(upcoming) if upcoming else None

    def get_dividend_yield(self, ticker: str, spot: float) -> float:
        """Trailing 12-month dividends divided by spot, used as the yield q."""
        try:
            dividends = yf.Ticker(ticker).dividends
        except Exception:
            return 0.0
        if dividends is None or dividends.empty or spot <= 0:
            return 0.0
        one_year_ago = pd.Timestamp(now_et() - timedelta(days=365))
        trailing = dividends[dividends.index >= one_year_ago].sum()
        return float(trailing / spot)

    def get_risk_free_rate(self) -> float | None:
        try:
            closes = yf.Ticker(TREASURY_BILL_SYMBOL).history(period="5d")["Close"].dropna()
        except Exception:
            return None
        if closes.empty:
            return None
        return float(closes.iloc[-1]) / 100.0
