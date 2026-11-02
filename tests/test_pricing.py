"""Tests for pricing.py against textbook values and model identities.

Reference values come from John C. Hull, "Options, Futures, and Other
Derivatives", whose worked examples are the standard check for a
Black-Scholes implementation.
"""

import math

import numpy as np
import pytest

from pricing import DAYS_PER_YEAR, bs_price, greeks, implied_volatility


# --- Textbook prices -------------------------------------------------------

def test_hull_call_and_put_prices():
    # Hull: S=42, K=40, r=10%, sigma=20%, T=6 months -> c = 4.76, p = 0.81
    assert bs_price(42, 40, 0.5, 0.10, 0.20, "call") == pytest.approx(4.76, abs=0.005)
    assert bs_price(42, 40, 0.5, 0.10, 0.20, "put") == pytest.approx(0.81, abs=0.005)


def test_hull_index_option_with_dividend_yield():
    # Hull: S=930, K=900, r=8%, q=3%, sigma=20%, T=2 months -> c = 51.83
    price = bs_price(930, 900, 2 / 12, 0.08, 0.20, "call", q=0.03)
    assert price == pytest.approx(51.83, abs=0.01)


# --- Textbook Greeks -------------------------------------------------------
# Hull's Greeks example: S=49, K=50, r=5%, sigma=20%, T=20 weeks.
# Hull quotes theta per year and vega/rho per 1.00 change in sigma/r;
# this module reports theta per day and vega/rho per 1 point, so we rescale.

HULL_GREEKS_INPUTS = dict(S=49, K=50, T=20 / 52, r=0.05, sigma=0.20)


def test_hull_call_greeks():
    g = greeks(**HULL_GREEKS_INPUTS, option_type="call")
    assert g["delta"] == pytest.approx(0.522, abs=0.001)
    assert g["gamma"] == pytest.approx(0.066, abs=0.001)
    assert g["theta"] * DAYS_PER_YEAR == pytest.approx(-4.31, abs=0.01)
    assert g["vega"] * 100 == pytest.approx(12.1, abs=0.05)
    assert g["rho"] * 100 == pytest.approx(8.91, abs=0.01)


def test_hull_call_price_for_greeks_example():
    assert bs_price(**HULL_GREEKS_INPUTS, option_type="call") == pytest.approx(2.40, abs=0.005)


# --- Model identities ------------------------------------------------------

@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100, 100, 1.0, 0.05, 0.20, 0.0),
        (100, 80, 0.25, 0.04, 0.45, 0.01),
        (50, 65, 0.05, 0.03, 0.90, 0.0),
        (250, 240, 2.0, 0.00, 0.15, 0.02),
    ],
)
def test_put_call_parity(S, K, T, r, sigma, q):
    # C - P = S e^(-qT) - K e^(-rT)
    call = bs_price(S, K, T, r, sigma, "call", q)
    put = bs_price(S, K, T, r, sigma, "put", q)
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_match_finite_differences(option_type):
    S, K, T, r, sigma, q = 100, 105, 0.4, 0.045, 0.32, 0.01
    g = greeks(S, K, T, r, sigma, option_type, q)

    def price(s=S, t=T, vol=sigma, rate=r):
        return bs_price(s, K, t, rate, vol, option_type, q)

    h = 0.01
    assert g["delta"] == pytest.approx((price(s=S + h) - price(s=S - h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx(
        (price(s=S + h) - 2 * price() + price(s=S - h)) / h**2, abs=1e-4
    )
    # Vega and rho per 1 point = price change for a 0.01 move in sigma / r.
    assert g["vega"] == pytest.approx((price(vol=sigma + 1e-4) - price(vol=sigma - 1e-4)) / 2e-4 / 100, abs=1e-6)
    assert g["rho"] == pytest.approx((price(rate=r + 1e-4) - price(rate=r - 1e-4)) / 2e-4 / 100, abs=1e-6)
    # Theta per day = price change as one day passes (T shrinks).
    one_day = 1 / DAYS_PER_YEAR
    assert g["theta"] == pytest.approx(price(t=T - one_day) - price(), abs=1e-3)


# --- Implied volatility ----------------------------------------------------

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [70, 90, 100, 110, 140])
@pytest.mark.parametrize("sigma", [0.08, 0.25, 0.60, 1.50])
def test_implied_vol_recovers_input_vol(option_type, K, sigma):
    S, T, r, q = 100, 0.3, 0.045, 0.005
    price = bs_price(S, K, T, r, sigma, option_type, q)
    # Deep ITM/OTM at low vol: price is ~all intrinsic (or ~0), vega ~0, and
    # the price carries no information about vol. Real quotes like this are
    # exactly the ones the surface skips.
    if greeks(S, K, T, r, sigma, option_type, q)["vega"] < 1e-4:
        pytest.skip("no vega: price is insensitive to vol")
    assert implied_volatility(price, S, K, T, r, option_type, q) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_matches_hull_example():
    assert implied_volatility(4.7594, 42, 40, 0.5, 0.10, "call") == pytest.approx(0.20, abs=1e-4)


@pytest.mark.parametrize(
    "price, S, K, T",
    [
        (0.0, 100, 100, 0.5),     # zero premium (e.g. zero bid and ask)
        (-1.0, 100, 100, 0.5),    # negative premium
        (5.0, 100, 90, 0.5),      # call priced below intrinsic value
        (150.0, 100, 100, 0.5),   # call priced above the stock price
        (5.0, 100, 100, 0.0),     # already expired
        (math.nan, 100, 100, 0.5),
    ],
)
def test_implied_vol_returns_nan_for_impossible_prices(price, S, K, T):
    assert math.isnan(implied_volatility(price, S, K, T, 0.05, "call"))


# --- Edge cases and vectorization ----------------------------------------

def test_expired_option_is_worth_intrinsic_value():
    assert bs_price(110, 100, 0.0, 0.05, 0.3, "call") == pytest.approx(10.0)
    assert bs_price(110, 100, 0.0, 0.05, 0.3, "put") == pytest.approx(0.0)
    assert greeks(110, 100, 0.0, 0.05, 0.3, "call")["delta"] == 1.0
    assert greeks(90, 100, 0.0, 0.05, 0.3, "put")["delta"] == -1.0
    assert greeks(110, 100, 0.0, 0.05, 0.3, "call")["gamma"] == 0.0


def test_vectorized_inputs_match_scalar_calls():
    spots = np.array([80.0, 100.0, 120.0])
    times = np.array([[0.1], [0.5]])  # broadcasts to a 2 x 3 grid
    grid = bs_price(spots, 100, times, 0.04, 0.3, "put")
    assert grid.shape == (2, 3)
    assert grid[1, 2] == pytest.approx(bs_price(120.0, 100, 0.5, 0.04, 0.3, "put"))


def test_scalar_inputs_return_plain_floats():
    assert isinstance(bs_price(100, 100, 1, 0.05, 0.2), float)
    assert isinstance(greeks(100, 100, 1, 0.05, 0.2)["delta"], float)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0.2, "straddle")
