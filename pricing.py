"""Black-Scholes-Merton pricing, Greeks, and implied volatility.

Conventions used everywhere in this project:
- ``T`` is time to expiry in years (calendar days / 365).
- ``r`` (risk-free rate), ``q`` (continuous dividend yield) and ``sigma``
  (volatility) are decimals: 0.05 means 5%.
- Theta is reported per calendar day, vega and rho per 1 percentage point.

``bs_price`` and ``greeks`` accept scalars or numpy arrays (broadcast together),
which lets the risk view reprice a whole grid of spot/date scenarios at once.
US equity options are American-style; Black-Scholes treats them as European,
which is a close approximation except for deep ITM puts and calls ahead of a
large dividend.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

DAYS_PER_YEAR = 365.0

# Search bounds for the IV solver: 0.01% to 500% annualized volatility.
IV_LOWER = 1e-4
IV_UPPER = 5.0


def _is_call(option_type: str) -> bool:
    kind = option_type.lower()
    if kind not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return kind == "call"


def _as_output(values: np.ndarray):
    """Return a plain float for scalar inputs, otherwise the array."""
    return float(values) if np.ndim(values) == 0 else values


def _prepare(S, K, T, r, sigma, q):
    """Broadcast inputs and split them into 'live' and 'expired / zero-vol' cases.

    The closed-form formulas divide by sigma * sqrt(T), so we substitute safe
    placeholder values where that is zero and select the correct result later.
    """
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    )
    live = (T > 0) & (sigma > 0)
    T_safe = np.where(live, T, 1.0)
    sigma_safe = np.where(live, sigma, 1.0)

    vol_sqrt_t = sigma_safe * np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe**2) * T_safe) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return S, K, T, r, sigma_safe, q, T_safe, live, d1, d2


def bs_price(S, K, T, r, sigma, option_type: str = "call", q=0.0):
    """Black-Scholes-Merton price of a European call or put.

    Call = S e^(-qT) N(d1) - K e^(-rT) N(d2)
    Put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

    At expiry (T <= 0) or with zero volatility the option is worth its
    discounted intrinsic value.
    """
    is_call = _is_call(option_type)
    S, K, T, r, _, q, _, live, d1, d2 = _prepare(S, K, T, r, sigma, q)

    disc_spot = S * np.exp(-q * np.maximum(T, 0))
    disc_strike = K * np.exp(-r * np.maximum(T, 0))

    if is_call:
        model = disc_spot * norm.cdf(d1) - disc_strike * norm.cdf(d2)
        intrinsic = np.maximum(disc_spot - disc_strike, 0.0)
    else:
        model = disc_strike * norm.cdf(-d2) - disc_spot * norm.cdf(-d1)
        intrinsic = np.maximum(disc_strike - disc_spot, 0.0)

    return _as_output(np.where(live, model, intrinsic))


def greeks(S, K, T, r, sigma, option_type: str = "call", q=0.0) -> dict:
    """Delta, gamma, theta, vega and rho for one option (per 1 share).

    Units: theta in $ per calendar day, vega in $ per 1 vol point,
    rho in $ per 1 percentage point of rates. Multiply by 100 for one
    standard equity contract.
    """
    is_call = _is_call(option_type)
    S, K, T, r, sigma, q, T_safe, live, d1, d2 = _prepare(S, K, T, r, sigma, q)

    div_disc = np.exp(-q * T_safe)
    rate_disc = np.exp(-r * T_safe)
    pdf_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(T_safe)

    gamma = div_disc * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * div_disc * pdf_d1 * sqrt_t / 100.0
    decay = -S * div_disc * pdf_d1 * sigma / (2.0 * sqrt_t)

    if is_call:
        delta = div_disc * norm.cdf(d1)
        theta = decay - r * K * rate_disc * norm.cdf(d2) + q * S * div_disc * norm.cdf(d1)
        rho = K * T_safe * rate_disc * norm.cdf(d2) / 100.0
        expired_delta = (S > K).astype(float)
    else:
        delta = -div_disc * norm.cdf(-d1)
        theta = decay + r * K * rate_disc * norm.cdf(-d2) - q * S * div_disc * norm.cdf(-d1)
        rho = -K * T_safe * rate_disc * norm.cdf(-d2) / 100.0
        expired_delta = -(S < K).astype(float)

    # Expired or zero-vol options behave like stock (ITM) or nothing (OTM).
    return {
        "delta": _as_output(np.where(live, delta, expired_delta)),
        "gamma": _as_output(np.where(live, gamma, 0.0)),
        "theta": _as_output(np.where(live, theta / DAYS_PER_YEAR, 0.0)),
        "vega": _as_output(np.where(live, vega, 0.0)),
        "rho": _as_output(np.where(live, rho, 0.0)),
    }


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Solve for the volatility that makes the Black-Scholes price equal ``price``.

    Uses Brent's method on [0.01%, 500%]. Returns NaN instead of raising when
    no sensible answer exists: a non-positive price, an expired option, or a
    price outside the no-arbitrage bounds (below intrinsic value or above the
    maximum an option can be worth). Callers treat NaN as "skip this quote".
    """
    is_call = _is_call(option_type)
    inputs = (price, S, K, T, r, q)
    if not all(math.isfinite(x) for x in inputs) or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return math.nan

    disc_spot = S * math.exp(-q * T)
    disc_strike = K * math.exp(-r * T)
    if is_call:
        floor, ceiling = max(disc_spot - disc_strike, 0.0), disc_spot
    else:
        floor, ceiling = max(disc_strike - disc_spot, 0.0), disc_strike
    if not floor < price < ceiling:
        return math.nan

    def pricing_error(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type, q) - price

    # Price must sit between the model price at the lowest and highest vol.
    if pricing_error(IV_LOWER) > 0 or pricing_error(IV_UPPER) < 0:
        return math.nan

    return brentq(pricing_error, IV_LOWER, IV_UPPER, xtol=1e-10, maxiter=200)
