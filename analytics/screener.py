"""Cash-secured put screener.

Selling a cash-secured put means collecting a premium today in exchange for
agreeing to buy 100 shares at the strike if the stock is below it at expiry.
The broker holds strike x 100 in cash as collateral. The screener ranks
puts by the return that premium earns on the collateral, annualized so that
a 7-day and a 45-day trade can be compared:

    annualized return = (premium / (strike x 100)) x (365 / DTE)

with premium = bid/ask mid x 100. Delta (absolute) is a rough probability
that the put finishes in the money, i.e. that you end up buying the shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from analytics.surface import add_delta, add_implied_vol, price_chain

CONTRACT_MULTIPLIER = 100


@dataclass(frozen=True)
class ScreenerFilters:
    max_collateral: float = 2_000.0
    min_abs_delta: float = 0.15
    max_abs_delta: float = 0.35
    min_open_interest: int = 100
    max_spread_pct: float = 0.15
    include_earnings: bool = False


# Filter rules in the order they are reported: (column, short label, hover text).
RULES = [
    ("fits_collateral", "collateral", lambda row: f"Needs ${row.collateral:,.0f} collateral"),
    ("delta_in_range", "delta", lambda row: f"Delta {row.delta:.2f} outside range"),
    ("enough_open_interest", "open interest", lambda row: f"Open interest {row.open_interest:,.0f} too low"),
    ("tight_spread", "spread", lambda row: f"Spread {row.spread_pct:.0%} of mid too wide"),
    ("earnings_ok", "earnings", lambda row: "Expires after earnings"),
]
CHECKS = [column for column, _, _ in RULES]


def put_candidates(
    chain: pd.DataFrame, ticker: str, spot: float, r: float, q: float, now: datetime, after_hours: bool
) -> pd.DataFrame:
    """Every put with a usable price, plus IV, delta and the return metrics.

    This is the slow part (it may solve IVs), so it is kept separate from the
    cheap filter step and can be cached independently of the filter settings.
    """
    priced = price_chain(chain, spot, now, after_hours)
    puts = priced[(priced["option_type"] == "put") & priced["price"].notna() & (priced["dte"] >= 1)]
    if puts.empty:
        return pd.DataFrame()
    puts = add_delta(add_implied_vol(puts, spot, r, q), spot, r, q)

    return puts.assign(
        ticker=ticker,
        spot=spot,
        collateral=puts["strike"] * CONTRACT_MULTIPLIER,
        premium=puts["price"] * CONTRACT_MULTIPLIER,
        annualized_return=(puts["price"] / puts["strike"]) * (365 / puts["dte"]),
        breakeven=puts["strike"] - puts["price"],
        pct_otm=(spot - puts["strike"]) / spot,
        abs_delta=puts["delta"].abs(),
    )


def apply_filters(
    candidates: pd.DataFrame, filters: ScreenerFilters, earnings_dates: dict[str, date | None]
) -> pd.DataFrame:
    """Add one pass/fail column per rule, plus ``passes``, ``fail_rule`` and ``fail_reason``."""
    if candidates.empty:
        return candidates
    df = candidates.copy()
    next_earnings = df["ticker"].map(earnings_dates)

    df["fits_collateral"] = df["collateral"] <= filters.max_collateral
    df["delta_in_range"] = df["abs_delta"].between(filters.min_abs_delta, filters.max_abs_delta)
    df["enough_open_interest"] = df["open_interest"] >= filters.min_open_interest
    # Last-trade prices (market closed) have no spread to check.
    df["tight_spread"] = (df["spread_pct"] <= filters.max_spread_pct) | (df["price_source"] == "last")
    # An option expiring on the earnings date counts as before it (companies
    # report outside market hours), matching the surface view.
    df["spans_earnings"] = [
        pd.notna(earnings) and expiry > earnings for expiry, earnings in zip(df["expiry"], next_earnings)
    ]
    df["earnings_ok"] = filters.include_earnings | ~df["spans_earnings"]

    df["passes"] = df[CHECKS].all(axis=1)
    first_failures = [
        next(((label, describe(row)) for column, label, describe in RULES if not getattr(row, column)), ("", ""))
        for row in df.itertuples(index=False)
    ]
    df["fail_rule"] = [label for label, _ in first_failures]
    df["fail_reason"] = [reason for _, reason in first_failures]
    return df


def heatmap_rows(screened: pd.DataFrame) -> pd.DataFrame:
    """Ticker/strike rows where at least one expiry passes every filter.

    The heat map then shows what is actually tradeable, with the near misses
    in those same rows greyed out, instead of hundreds of irrelevant strikes."""
    if screened.empty:
        return screened
    keys = screened.loc[screened["passes"], ["ticker", "strike"]].drop_duplicates()
    return screened.merge(keys, on=["ticker", "strike"])


def top_candidates(screened: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if screened.empty:
        return screened
    passing = screened[screened["passes"]]
    return passing.sort_values("annualized_return", ascending=False).head(n).reset_index(drop=True)


def ticker_summary(screened: pd.DataFrame, filters: ScreenerFilters) -> pd.DataFrame:
    """One line per ticker: how many puts pass, or the main reason none do."""
    rows = []
    for ticker, group in screened.groupby("ticker", sort=False):
        n_pass = int(group["passes"].sum())
        if n_pass:
            note = f"{n_pass} put{'s' if n_pass > 1 else ''} pass every filter"
        elif not group["fits_collateral"].any():
            note = (
                f"Lowest strike needs ${group['collateral'].min():,.0f} collateral "
                f"(cap ${filters.max_collateral:,.0f})"
            )
        else:
            blocker = group.loc[group["fits_collateral"], "fail_rule"].mode().iloc[0]
            note = f"Affordable strikes exist, but fail mostly on {blocker}"
        rows.append({"ticker": ticker, "spot": group["spot"].iloc[0], "passing": n_pass, "note": note})
    return pd.DataFrame(rows, columns=["ticker", "spot", "passing", "note"])


def cell_grid(rows: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot to (ticker, strike) rows x expiry columns: tickers in watchlist
    order, strikes high to low within each ticker, like a broker's put chain."""
    order = {ticker: i for i, ticker in enumerate(dict.fromkeys(rows["ticker"]))}
    grid = rows.pivot_table(index=["ticker", "strike"], columns="expiry", values=value, aggfunc="first")
    grid = grid.reset_index()
    grid = grid.sort_values(
        ["ticker", "strike"], key=lambda col: col.map(order) if col.name == "ticker" else -col
    )
    return grid.set_index(["ticker", "strike"])
