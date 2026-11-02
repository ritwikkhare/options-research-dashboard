"""Surface pipeline tests on a synthetic chain priced at a known 30% vol."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from analytics.surface import atm_term_structure, build_surface, common_strikes, select_side, surface_grid
from market_hours import ET, years_to_expiry
from pricing import bs_price
from providers import CHAIN_COLUMNS

NOW = datetime(2026, 11, 2, 11, 0, tzinfo=ET)  # Monday, market open
SPOT, R, Q, VOL = 100.0, 0.04, 0.0, 0.30
EXPIRIES = [date(2026, 11, 20), date(2026, 12, 18)]
STRIKES = [80, 90, 95, 100, 105, 110, 120]


def synthetic_chain(provider_iv=VOL, half_spread=0.02) -> pd.DataFrame:
    """Every contract quoted at fair value +/- 2%, so all pass the spread filter."""
    rows = []
    for expiry in EXPIRIES:
        T = years_to_expiry(expiry, NOW)
        for strike in STRIKES:
            for option_type in ("call", "put"):
                fair = bs_price(SPOT, strike, T, R, VOL, option_type, Q)
                rows.append(
                    {
                        "contract": f"X{expiry:%y%m%d}{option_type[0].upper()}{strike}",
                        "option_type": option_type,
                        "expiry": expiry,
                        "strike": float(strike),
                        "bid": fair * (1 - half_spread),
                        "ask": fair * (1 + half_spread),
                        "last": fair,
                        "volume": 50.0,
                        "open_interest": 500.0,
                        "provider_iv": provider_iv,
                        "last_trade": pd.Timestamp("2026-10-30 20:00", tz="UTC"),
                    }
                )
    return pd.DataFrame(rows)[CHAIN_COLUMNS]


def build(chain, after_hours=False):
    return build_surface(chain, SPOT, R, Q, NOW, after_hours)


def test_good_provider_iv_is_used_as_is():
    surface = build(synthetic_chain(provider_iv=0.31))
    assert (surface["iv_source"] == "provider").all()
    assert surface["iv"].eq(0.31).all()


def test_junk_provider_iv_is_recomputed_from_the_mid():
    # yfinance reports 0.00001 for many contracts; the solver should recover ~30%.
    surface = build(synthetic_chain(provider_iv=0.00001))
    assert (surface["iv_source"] == "computed").all()
    near_atm = surface[(surface["strike"] - SPOT).abs() <= 10]
    assert near_atm["iv"].to_numpy() == pytest.approx(VOL, abs=0.01)


def test_illiquid_contracts_are_skipped():
    chain = synthetic_chain()
    chain.loc[0, ["bid", "ask"]] = 0.0                             # no quote
    chain.loc[1, ["open_interest", "volume"]] = 0.0                # nobody trades it
    chain.loc[2, ["bid", "ask"]] = [0.10, 5.00]                    # absurd spread
    surface = build(chain)
    assert not surface["contract"].isin(chain.loc[:2, "contract"]).any()
    assert len(surface) == len(chain) - 3


def test_after_hours_falls_back_to_recent_last_trade():
    chain = synthetic_chain(provider_iv=0.00001)
    chain[["bid", "ask"]] = 0.0  # what yfinance returns overnight
    assert build(chain, after_hours=False).empty
    surface = build(chain, after_hours=True)
    assert (surface["price_source"] == "last").all()
    assert (surface["iv_source"] == "computed").all()


def test_stale_last_trade_is_not_used_after_hours():
    chain = synthetic_chain()
    chain[["bid", "ask"]] = 0.0
    chain["last_trade"] = pd.Timestamp("2026-10-01", tz="UTC")
    assert build(chain, after_hours=True).empty


def test_otm_side_uses_puts_below_and_calls_above_spot():
    otm = select_side(build(synthetic_chain()), "otm")
    assert (otm.loc[otm["option_type"] == "put", "strike"] < SPOT).all()
    assert (otm.loc[otm["option_type"] == "call", "strike"] >= SPOT).all()
    assert not otm.duplicated(["expiry", "strike"]).any()


def test_delta_is_added_with_correct_signs():
    surface = build(synthetic_chain())
    assert (surface.loc[surface["option_type"] == "call", "delta"] > 0).all()
    assert (surface.loc[surface["option_type"] == "put", "delta"] < 0).all()


def test_surface_grid_is_strikes_by_expiries():
    grid = surface_grid(select_side(build(synthetic_chain()), "otm"), "iv")
    assert list(grid.index) == sorted(STRIKES)
    assert list(grid.columns) == EXPIRIES


def test_common_strikes_drops_strikes_listed_in_few_expiries():
    contracts = select_side(build(synthetic_chain()), "otm")
    extra = contracts[contracts["strike"] == 105].head(1).assign(strike=102.5)  # weekly-only strike
    kept = common_strikes(pd.concat([contracts, extra]), min_share=0.75)
    assert 102.5 not in kept["strike"].values
    assert set(kept["strike"]) == set(contracts["strike"])


def test_atm_term_structure_interpolates_to_spot():
    contracts = select_side(build(synthetic_chain()), "otm").copy()
    contracts["iv"] = np.where(contracts["strike"] < SPOT, 0.40, 0.20)
    # Strikes 95 (0.40) and 100 (0.20) bracket spot=100 exactly at 100.
    term = atm_term_structure(contracts, 97.5)
    assert term["atm_iv"].to_numpy() == pytest.approx([0.30, 0.30])
    assert list(term["expiry"]) == EXPIRIES


def test_atm_term_structure_skips_expiries_without_strikes_around_spot():
    contracts = select_side(build(synthetic_chain()), "call")
    assert atm_term_structure(contracts, 500.0).empty


def test_empty_chain_after_filters_returns_empty_frame():
    chain = synthetic_chain()
    chain[["bid", "ask"]] = 0.0
    surface = build(chain)
    assert surface.empty
    assert {"iv", "iv_source", "delta"} <= set(surface.columns)
