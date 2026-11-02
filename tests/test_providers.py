"""Offline tests for the provider layer (no network calls)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from providers import CHAIN_COLUMNS, DataProvider, DataProviderError, NoOptionsError, get_provider
from providers.yfinance_provider import YFinanceProvider, normalize_yf_chain

EXPIRY = date(2026, 11, 20)


def raw_yfinance_calls() -> pd.DataFrame:
    """Shape of yfinance's option_chain(...).calls, including its quirks."""
    return pd.DataFrame(
        {
            "contractSymbol": ["NVDA261120C00200000", "NVDA261120C00210000"],
            "lastTradeDate": pd.to_datetime(["2026-10-30 19:59:00", "2026-10-30 19:58:00"], utc=True),
            "strike": [200.0, 210.0],
            "lastPrice": [9.10, 4.85],
            "bid": [9.00, 0.0],
            "ask": [9.20, 0.0],
            "change": [0.0, 0.0],
            "percentChange": [0.0, 0.0],
            "volume": [1500.0, np.nan],  # NaN volume = no trades today
            "openInterest": [12000, 8000],
            "impliedVolatility": [0.41, 0.00001],
            "inTheMoney": [True, False],
            "contractSize": ["REGULAR", "REGULAR"],
            "currency": ["USD", "USD"],
        }
    )


def test_normalize_yf_chain_maps_to_shared_schema():
    chain = normalize_yf_chain(raw_yfinance_calls(), "call", EXPIRY)
    assert list(chain.columns) == CHAIN_COLUMNS
    assert (chain["option_type"] == "call").all()
    assert (chain["expiry"] == EXPIRY).all()
    assert chain.loc[0, "provider_iv"] == pytest.approx(0.41)
    assert chain.loc[1, "volume"] == 0  # missing volume becomes 0, not NaN


def test_normalize_yf_chain_handles_empty_input():
    chain = normalize_yf_chain(pd.DataFrame(), "put", EXPIRY)
    assert chain.empty
    assert list(chain.columns) == CHAIN_COLUMNS


class FakeProvider(DataProvider):
    """Returns a one-row chain, except for expiries listed in ``failing``."""

    def __init__(self, failing=()):
        self.failing = set(failing)

    def get_spot(self, ticker):
        return 100.0

    def get_expirations(self, ticker):
        return [date(2026, 11, 20), date(2026, 12, 18)]

    def get_chain(self, ticker, expiry):
        if expiry in self.failing:
            raise DataProviderError("boom")
        return normalize_yf_chain(raw_yfinance_calls().head(1), "call", expiry)


def test_get_chains_skips_expiries_that_fail():
    provider = FakeProvider(failing=[date(2026, 12, 18)])
    chains = provider.get_chains("NVDA", provider.get_expirations("NVDA"))
    assert chains["expiry"].unique().tolist() == [date(2026, 11, 20)]


def test_get_chains_raises_when_nothing_loads():
    provider = FakeProvider(failing=[date(2026, 11, 20), date(2026, 12, 18)])
    with pytest.raises(NoOptionsError):
        provider.get_chains("NVDA", provider.get_expirations("NVDA"))


def test_optional_methods_have_safe_defaults():
    provider = FakeProvider()
    assert provider.get_earnings_date("NVDA") is None
    assert provider.get_dividend_yield("NVDA", 100.0) == 0.0
    assert provider.get_risk_free_rate() is None


def test_get_provider_defaults_to_yfinance(monkeypatch):
    monkeypatch.delenv("DATA_PROVIDER", raising=False)
    monkeypatch.setattr("providers.load_dotenv", lambda: None)
    assert isinstance(get_provider(), YFinanceProvider)


def test_get_provider_rejects_unknown_names():
    with pytest.raises(DataProviderError):
        get_provider("bloomberg")


def test_tradier_requires_a_token(monkeypatch):
    monkeypatch.setenv("TRADIER_TOKEN", "")
    monkeypatch.setattr("providers.load_dotenv", lambda: None)
    with pytest.raises(DataProviderError, match="TRADIER_TOKEN"):
        get_provider("tradier")
