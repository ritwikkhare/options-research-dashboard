"""Risk view tests: parsing, position Greeks, collateral, flags and the P&L grid."""

import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.risk import (
    grid_dates,
    lookup_iv,
    parse_positions,
    pnl_grid,
    portfolio_totals,
    value_positions,
)
from market_hours import ET
from pricing import bs_price, greeks

NOW = datetime(2026, 11, 2, 11, 0, tzinfo=ET)
TODAY = NOW.date()
R = 0.04
SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample_positions.csv"


def positions(*rows) -> pd.DataFrame:
    columns = ["ticker", "type", "side", "strike", "expiry", "quantity", "entry_price"]
    parsed, problems = parse_positions(pd.DataFrame(rows, columns=columns), TODAY)
    assert not problems, problems
    return parsed


def value(parsed, spot=100.0, iv=0.30):
    tickers = parsed["ticker"].unique()
    return value_positions(
        parsed, {t: spot for t in tickers}, {t: 0.0 for t in tickers}, [iv] * len(parsed), R, NOW
    )


# --- Parsing ---------------------------------------------------------------

def test_sample_csv_parses_cleanly():
    parsed, problems = parse_positions(pd.read_csv(SAMPLE_CSV), TODAY)
    assert problems == []
    assert len(parsed) == 4
    assert parsed.loc[parsed["type"] == "stock", "strike"].isna().all()


def test_bad_rows_are_reported_and_skipped():
    raw = pd.DataFrame(
        [
            ["SOFI", "put", "short", 15, "2026-11-20", 1, 0.40],     # fine
            ["SOFI", "straddle", "short", 15, "2026-11-20", 1, 0.40],  # bad type
            ["SOFI", "put", "short", 15, "2026-10-01", 1, 0.40],     # expired
            ["SOFI", "put", "short", None, "2026-11-20", 1, 0.40],   # no strike
            ["SOFI", "put", "short", 15, "2026-11-20", -1, 0.40],    # negative qty
        ],
        columns=["Ticker", "Type", "Side", "Strike", "Expiry", "Quantity", "Entry Price"],
    )
    parsed, problems = parse_positions(raw, TODAY)
    assert len(parsed) == 1
    assert len(problems) == 4
    assert "Row 3" in problems[1] and "expired" in problems[1]


def test_missing_columns_are_reported():
    parsed, problems = parse_positions(pd.DataFrame({"ticker": ["SOFI"]}), TODAY)
    assert parsed.empty
    assert "Missing column" in problems[0]


# --- Valuation and Greeks --------------------------------------------------

def test_short_put_greeks_are_negated_and_scaled_by_100():
    parsed = positions(["SOFI", "put", "short", 95, "2026-11-20", 2, 1.00])
    row = value(parsed).iloc[0]
    T = (datetime(2026, 11, 20, 16, tzinfo=ET) - NOW).total_seconds() / (365 * 86400)
    per_share = greeks(100, 95, T, R, 0.30, "put")
    assert row["delta"] == pytest.approx(-200 * per_share["delta"])
    assert row["delta"] > 0 and row["theta"] > 0 and row["vega"] < 0  # short premium
    assert row["pnl"] == pytest.approx((bs_price(100, 95, T, R, 0.30, "put") - 1.00) * -200)


def test_stock_position_has_delta_equal_to_shares():
    row = value(positions(["F", "stock", "long", None, None, 100, 90.0])).iloc[0]
    assert row["delta"] == 100
    assert row["pnl"] == pytest.approx((100 - 90) * 100)
    assert row["max_loss"] == pytest.approx(9_000)


def test_collateral_and_max_loss():
    valued = value(positions(
        ["SOFI", "put", "short", 15, "2026-11-20", 1, 0.40],
        ["SOFI", "call", "long", 18, "2026-11-20", 2, 0.25],
    ), spot=16.0)
    put, call = valued.iloc[0], valued.iloc[1]
    assert put["collateral"] == 1_500
    assert put["max_loss"] == pytest.approx((15 - 0.40) * 100)
    assert call["collateral"] == 0
    assert call["max_loss"] == pytest.approx(0.25 * 200)
    assert portfolio_totals(valued)["collateral"] == 1_500


def test_covered_call_is_recognized_and_naked_call_flagged():
    covered = value(positions(
        ["F", "stock", "long", None, None, 100, 11.8],
        ["F", "call", "short", 13, "2026-12-18", 1, 0.25],
    ), spot=12.0)
    assert covered.iloc[1]["covered"]
    assert covered.iloc[1]["max_loss"] == 0
    assert covered.iloc[1]["flag"] == ""

    naked = value(positions(["F", "call", "short", 13, "2026-12-18", 1, 0.25]), spot=12.0)
    assert math.isinf(naked.iloc[0]["max_loss"])
    assert "Uncovered" in naked.iloc[0]["flag"]


@pytest.mark.parametrize(
    "spot, expiry, expected",
    [
        (14.0, "2026-11-06", "assignment likely"),  # ITM, 4 days left
        (14.0, "2026-12-18", "In the money"),       # ITM, plenty of time
        (15.5, "2026-11-06", "Within 5%"),          # 3.3% above strike, 4 days left
        (15.5, "2026-12-18", ""),                   # near strike but not expiring soon
        (18.0, "2026-11-06", ""),                   # comfortably OTM
    ],
)
def test_short_put_flags(spot, expiry, expected):
    flag = value(positions(["SOFI", "put", "short", 15, expiry, 1, 0.40]), spot=spot).iloc[0]["flag"]
    assert (expected in flag) if expected else flag == ""


# --- IV lookup -------------------------------------------------------------

def test_lookup_iv_interpolates_on_nearest_expiry():
    contracts = pd.DataFrame({
        "expiry": [date(2026, 11, 20)] * 2 + [date(2026, 12, 18)] * 2,
        "strike": [90.0, 110.0, 90.0, 110.0],
        "iv": [0.40, 0.20, 0.50, 0.30],
    })
    assert lookup_iv(contracts, 100.0, date(2026, 11, 21)) == pytest.approx(0.30)
    assert lookup_iv(contracts, 200.0, date(2026, 12, 18)) == pytest.approx(0.30)  # flat beyond range
    assert math.isnan(lookup_iv(contracts.iloc[0:0], 100.0, date(2026, 11, 20)))


# --- P&L grid --------------------------------------------------------------

def test_grid_today_at_zero_move_matches_current_pnl():
    valued = value(positions(
        ["SOFI", "put", "short", 95, "2026-11-20", 1, 2.00],
        ["SOFI", "stock", "long", None, None, 50, 98.0],
    ))
    grid = pnl_grid(valued, R, NOW)
    assert grid.shape == (41, len(grid_dates(valued, TODAY)))
    assert grid.loc[0.0, TODAY] == pytest.approx(portfolio_totals(valued)["pnl"])


def test_short_put_at_expiry_is_intrinsic():
    valued = value(positions(["SOFI", "put", "short", 95, "2026-11-20", 1, 2.00]))
    grid = pnl_grid(valued, R, NOW)
    expiry = date(2026, 11, 20)
    # -20% move: stock at 80, put worth 15, we received 2 -> lose 13 per share.
    assert grid.loc[-0.20, expiry] == pytest.approx(-1_300)
    # Stock unchanged at 100: put expires worthless, we keep the 2 premium.
    assert grid.loc[0.0, expiry] == pytest.approx(200)


def test_iv_shift_helps_long_options_and_hurts_short_ones():
    long_call = value(positions(["X", "call", "long", 100, "2026-12-18", 1, 3.0]))
    short_put = value(positions(["X", "put", "short", 100, "2026-12-18", 1, 3.0]))
    assert pnl_grid(long_call, R, NOW, iv_shift=0.10).iloc[20, 0] > pnl_grid(long_call, R, NOW).iloc[20, 0]
    assert pnl_grid(short_put, R, NOW, iv_shift=0.10).iloc[20, 0] < pnl_grid(short_put, R, NOW).iloc[20, 0]


def test_grid_dates_always_include_each_expiry():
    valued = value(positions(
        ["X", "put", "short", 95, "2026-11-20", 1, 2.0],
        ["X", "call", "long", 110, "2027-06-18", 1, 5.0],
    ))
    dates = grid_dates(valued, TODAY, max_columns=45)
    assert date(2026, 11, 20) in dates and date(2027, 6, 18) in dates
    assert len(dates) <= 47
    assert dates == sorted(dates)


def test_stock_only_portfolio_gets_a_30_day_grid():
    valued = value(positions(["F", "stock", "long", None, None, 100, 90.0]))
    grid = pnl_grid(valued, R, NOW)
    assert grid.shape[1] == 31
    assert np.allclose(grid.loc[0.10], (110 - 90) * 100)
