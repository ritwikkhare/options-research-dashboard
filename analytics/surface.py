"""Implied volatility surface: clean a raw chain, fill in IV, shape it for charts.

Pipeline (``build_surface``):
1. Price each contract. During market hours that is the bid/ask mid. Outside
   market hours free feeds zero out bid/ask, so we fall back to the last trade
   price if it is recent, and label it as such.
2. Drop illiquid quotes (no price, wide spread, no open interest or volume,
   strikes far from spot): plotting them would add noise, not information.
3. Take the provider's IV where it is believable; otherwise solve for IV from
   our price with Black-Scholes and mark the contract as "computed".
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from market_hours import days_to_expiry, years_to_expiry
from pricing import greeks, implied_volatility

# Provider IVs outside this range (1% to 500%) are treated as junk.
MIN_IV, MAX_IV = 0.01, 5.0
# Defaults for what counts as a usable quote.
MAX_SPREAD_PCT = 0.50        # bid/ask spread as a share of the mid
MIN_OPEN_INTEREST = 10
STRIKE_WINDOW = 0.40         # keep strikes within +/-40% of spot
MAX_LAST_TRADE_AGE = timedelta(days=5)  # covers a weekend plus a holiday

SIDES = {
    "otm": "OTM only (puts below spot, calls above)",
    "call": "Calls",
    "put": "Puts",
}


def price_chain(chain: pd.DataFrame, spot: float, now: datetime, after_hours: bool) -> pd.DataFrame:
    """Add time to expiry, a usable price, spread and moneyness columns."""
    df = chain.copy()

    expiries = df["expiry"].unique()
    dte = {e: days_to_expiry(e, now) for e in expiries}
    years = {e: years_to_expiry(e, now) for e in expiries}
    df["dte"] = df["expiry"].map(dte)
    df["T"] = df["expiry"].map(years)

    has_quote = (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])
    mid = (df["bid"] + df["ask"]) / 2
    df["mid"] = mid.where(has_quote)
    df["spread_pct"] = ((df["ask"] - df["bid"]) / mid).where(has_quote)

    trade_age = pd.Timestamp(now).tz_convert("UTC") - df["last_trade"]
    use_last = after_hours & ~has_quote & (df["last"] > 0) & (trade_age <= MAX_LAST_TRADE_AGE)

    df["price"] = df["mid"].where(has_quote, df["last"].where(use_last))
    df["price_source"] = np.select([has_quote, use_last], ["mid", "last"], default="none")
    df["moneyness"] = df["strike"] / spot
    df["otm"] = ((df["option_type"] == "put") & (df["strike"] < spot)) | (
        (df["option_type"] == "call") & (df["strike"] >= spot)
    )
    return df


def liquid_mask(
    df: pd.DataFrame,
    min_open_interest: int = MIN_OPEN_INTEREST,
    max_spread_pct: float = MAX_SPREAD_PCT,
) -> pd.Series:
    """True for contracts with a price we trust enough to plot."""
    has_price = df["price"].notna() & (df["T"] > 0)
    has_interest = (df["open_interest"] >= min_open_interest) | (df["volume"] > 0)
    tight_enough = (df["spread_pct"] <= max_spread_pct) | (df["price_source"] == "last")
    near_spot = (df["moneyness"] - 1).abs() <= STRIKE_WINDOW
    return has_price & has_interest & tight_enough & near_spot


def add_implied_vol(df: pd.DataFrame, spot: float, r: float, q: float) -> pd.DataFrame:
    """Add ``iv`` and ``iv_source`` ("provider" or "computed"); drop unsolvable rows."""
    df = df.copy()
    provider_ok = df["provider_iv"].between(MIN_IV, MAX_IV) & (df["price_source"] == "mid")

    computed = [
        implied_volatility(row.price, spot, row.strike, row.T, r, row.option_type, q)
        if not ok
        else np.nan
        for row, ok in zip(df.itertuples(index=False), provider_ok)
    ]
    df["iv"] = df["provider_iv"].where(provider_ok, computed)
    df["iv_source"] = np.where(provider_ok, "provider", "computed")
    return df[df["iv"].between(MIN_IV, MAX_IV)].reset_index(drop=True)


def add_delta(df: pd.DataFrame, spot: float, r: float, q: float) -> pd.DataFrame:
    df = df.copy()
    df["delta"] = np.nan
    for option_type in ("call", "put"):
        rows = df["option_type"] == option_type
        if rows.any():
            df.loc[rows, "delta"] = greeks(
                spot, df.loc[rows, "strike"].to_numpy(), df.loc[rows, "T"].to_numpy(),
                r, df.loc[rows, "iv"].to_numpy(), option_type, q,
            )["delta"]
    return df


def build_surface(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
    now: datetime,
    after_hours: bool,
    min_open_interest: int = MIN_OPEN_INTEREST,
    max_spread_pct: float = MAX_SPREAD_PCT,
) -> pd.DataFrame:
    """Raw chain -> liquid contracts with ``iv``, ``iv_source`` and ``delta``."""
    priced = price_chain(chain, spot, now, after_hours)
    liquid = priced[liquid_mask(priced, min_open_interest, max_spread_pct)]
    if liquid.empty:
        return liquid.assign(iv=pd.Series(dtype=float), iv_source="", delta=pd.Series(dtype=float))
    with_iv = add_implied_vol(liquid, spot, r, q)
    return add_delta(with_iv, spot, r, q)


def select_side(surface: pd.DataFrame, side: str) -> pd.DataFrame:
    """Contracts for one view: calls, puts, or OTM-only."""
    if side == "otm":
        return surface[surface["otm"]]
    if side in ("call", "put"):
        return surface[surface["option_type"] == side]
    raise ValueError(f"Unknown side {side!r}")


def common_strikes(contracts: pd.DataFrame, min_share: float = 0.5) -> pd.DataFrame:
    """Keep strikes listed in at least ``min_share`` of the expiries.

    Weekly expiries list in-between strikes (e.g. every $2.50) that monthlies
    don't. On a heat map those show up as mostly-empty stripes, so the
    surface keeps the shared strike grid; the smile still shows every strike.
    """
    n_expiries = contracts["expiry"].nunique()
    listed_in = contracts.groupby("strike")["expiry"].nunique()
    keep = listed_in[listed_in >= min_share * n_expiries].index
    return contracts[contracts["strike"].isin(keep)]


def surface_grid(contracts: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot one column into a strike (rows) x expiry (columns) grid."""
    return contracts.pivot_table(index="strike", columns="expiry", values=value, aggfunc="first").sort_index()


def smile(contracts: pd.DataFrame, expiry: date) -> pd.DataFrame:
    """IV by strike for one expiry."""
    return contracts[contracts["expiry"] == expiry].sort_values("strike")


def atm_term_structure(contracts: pd.DataFrame, spot: float) -> pd.DataFrame:
    """At-the-money IV per expiry, linearly interpolated to the spot price.

    An expiry is skipped unless it has strikes on both sides of spot, so the
    value is always an interpolation, never an extrapolation.
    """
    rows = []
    for expiry, group in contracts.groupby("expiry"):
        group = group.sort_values("strike")
        strikes, ivs = group["strike"].to_numpy(), group["iv"].to_numpy()
        if strikes.min() <= spot <= strikes.max():
            rows.append({"expiry": expiry, "dte": group["dte"].iloc[0], "atm_iv": np.interp(spot, strikes, ivs)})
    return pd.DataFrame(rows, columns=["expiry", "dte", "atm_iv"])
