"""Tradier data provider (stub): real-time quotes with a Tradier account.

To finish this provider, implement each method with a GET request and map
the JSON onto CHAIN_COLUMNS. Only market-data endpoints are used; this
project never calls Tradier's trading or account endpoints.

    Base URL   https://api.tradier.com/v1   (sandbox: https://sandbox.tradier.com/v1;
               sandbox quotes are delayed)
    Headers    Authorization: Bearer <TRADIER_TOKEN>, Accept: application/json

    get_spot          GET /markets/quotes?symbols=NVDA            -> quotes.quote.last
    get_expirations   GET /markets/options/expirations?symbol=NVDA -> expirations.date[]
    get_chain         GET /markets/options/chains?symbol=NVDA&expiration=YYYY-MM-DD&greeks=true
                      -> options.option[]: symbol, option_type, strike, bid, ask, last,
                         volume, open_interest, trade_date, greeks.mid_iv
    get_earnings_date GET /beta/markets/fundamentals/calendars?symbols=NVDA
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from providers.base import DataProvider

LIVE_URL = "https://api.tradier.com/v1"
SANDBOX_URL = "https://sandbox.tradier.com/v1"

NOT_IMPLEMENTED = "The Tradier provider is a stub. Set DATA_PROVIDER=yfinance or implement providers/tradier_provider.py."


class TradierProvider(DataProvider):
    name = "tradier"
    is_realtime = True

    def __init__(self, token: str, sandbox: bool = True):
        self.base_url = SANDBOX_URL if sandbox else LIVE_URL
        self._headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def get_spot(self, ticker: str) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED)

    def get_expirations(self, ticker: str) -> list[date]:
        raise NotImplementedError(NOT_IMPLEMENTED)

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        raise NotImplementedError(NOT_IMPLEMENTED)
