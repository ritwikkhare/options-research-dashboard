"""Screener tests on a synthetic put chain priced at a known 40% vol."""

from datetime import date, datetime

import pandas as pd
import pytest

from analytics.screener import (
    ScreenerFilters,
    apply_filters,
    cell_grid,
    heatmap_rows,
    put_candidates,
    ticker_summary,
    top_candidates,
)
from market_hours import ET, years_to_expiry
from pricing import bs_price
from providers import CHAIN_COLUMNS

NOW = datetime(2026, 11, 2, 11, 0, tzinfo=ET)
R, Q, VOL = 0.04, 0.0, 0.40
NOV, DEC = date(2026, 11, 20), date(2026, 12, 18)  # 18 and 46 DTE


def put_chain(spot: float, strikes, expiries=(NOV, DEC), open_interest=500.0) -> pd.DataFrame:
    rows = []
    for expiry in expiries:
        T = years_to_expiry(expiry, NOW)
        for strike in strikes:
            fair = bs_price(spot, strike, T, R, VOL, "put", Q)
            rows.append({
                "contract": f"P{expiry:%m%d}{strike}", "option_type": "put", "expiry": expiry,
                "strike": float(strike), "bid": fair * 0.98, "ask": fair * 1.02, "last": fair,
                "volume": 10.0, "open_interest": open_interest, "provider_iv": VOL,
                "last_trade": pd.Timestamp("2026-10-30 20:00", tz="UTC"),
            })
    return pd.DataFrame(rows)[CHAIN_COLUMNS]


def candidates(ticker="SOFI", spot=16.0, strikes=(12, 13, 14, 15, 16), **chain_kwargs):
    return put_candidates(put_chain(spot, strikes, **chain_kwargs), ticker, spot, R, Q, NOW, after_hours=False)


def test_annualized_return_formula():
    puts = candidates()
    row = puts[(puts["strike"] == 15) & (puts["expiry"] == NOV)].iloc[0]
    mid = row["price"]
    assert row["dte"] == 18
    assert row["annualized_return"] == pytest.approx((mid * 100) / (15 * 100) * (365 / 18))
    assert row["breakeven"] == pytest.approx(15 - mid)
    assert row["pct_otm"] == pytest.approx((16 - 15) / 16)
    assert row["collateral"] == 1500


def test_collateral_cap_rejects_expensive_strikes():
    screened = apply_filters(candidates("NVDA", 200.0, (180, 190)), ScreenerFilters(), {})
    assert not screened["passes"].any()
    assert (screened["fail_rule"] == "collateral").all()
    assert screened["fail_reason"].iloc[0].startswith("Needs $18,000")


def test_delta_range_filter():
    screened = apply_filters(candidates(), ScreenerFilters(min_abs_delta=0.15, max_abs_delta=0.35), {})
    passing = screened[screened["passes"]]
    assert not passing.empty
    assert passing["abs_delta"].between(0.15, 0.35).all()
    # The at-the-money 16 strike has |delta| near 0.5, so it must fail on delta.
    atm = screened[screened["strike"] == 16]
    assert (atm["fail_rule"] == "delta").all()


def test_open_interest_and_spread_filters():
    thin = apply_filters(candidates(open_interest=5.0), ScreenerFilters(min_abs_delta=0, max_abs_delta=1), {})
    assert (thin["fail_rule"] == "open interest").all()

    wide = candidates()
    wide["spread_pct"] = 0.40
    screened = apply_filters(wide, ScreenerFilters(min_abs_delta=0, max_abs_delta=1), {})
    assert (screened["fail_rule"] == "spread").all()


def test_earnings_rule_and_toggle():
    loose = dict(min_abs_delta=0, max_abs_delta=1)
    earnings = {"SOFI": date(2026, 12, 1)}  # between the Nov and Dec expiries
    screened = apply_filters(candidates(), ScreenerFilters(**loose), earnings)
    assert screened.loc[screened["expiry"] == NOV, "passes"].all()
    assert (screened.loc[screened["expiry"] == DEC, "fail_rule"] == "earnings").all()

    included = apply_filters(candidates(), ScreenerFilters(**loose, include_earnings=True), earnings)
    assert included["passes"].all()


def test_expiry_on_earnings_day_counts_as_before():
    screened = apply_filters(candidates(), ScreenerFilters(min_abs_delta=0, max_abs_delta=1), {"SOFI": NOV})
    assert screened.loc[screened["expiry"] == NOV, "passes"].all()


def test_top_candidates_sorted_by_return_and_capped():
    loose = ScreenerFilters(min_abs_delta=0, max_abs_delta=1)
    top = top_candidates(apply_filters(candidates(), loose, {}), n=3)
    assert len(top) == 3
    assert top["annualized_return"].is_monotonic_decreasing


def test_heatmap_rows_and_summary_across_watchlist():
    watchlist = pd.concat([candidates("NVDA", 200.0, (180, 190)), candidates()])
    screened = apply_filters(watchlist, ScreenerFilters(), {})
    rows = heatmap_rows(screened)
    assert set(rows["ticker"]) == {"SOFI"}  # no NVDA strike fits $2,000

    summary = ticker_summary(screened, ScreenerFilters()).set_index("ticker")
    assert summary.loc["NVDA", "passing"] == 0
    assert "collateral" in summary.loc["NVDA", "note"]
    assert summary.loc["SOFI", "passing"] > 0


def test_cell_grid_orders_tickers_then_strikes_descending():
    loose = ScreenerFilters(min_abs_delta=0, max_abs_delta=1, max_collateral=1e9)
    screened = apply_filters(pd.concat([candidates("SOFI"), candidates("F", 12.0, (10, 11))]), loose, {})
    grid = cell_grid(screened, "annualized_return")
    assert list(grid.index) == [("SOFI", s) for s in (16, 15, 14, 13, 12)] + [("F", 11), ("F", 10)]
    assert list(grid.columns) == [NOV, DEC]


def test_no_puts_returns_empty_frame():
    chain = put_chain(16.0, (15,))
    chain[["bid", "ask"]] = 0.0
    assert put_candidates(chain, "SOFI", 16.0, R, Q, NOW, after_hours=False).empty
