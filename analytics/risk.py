"""Portfolio risk: parse positions, value them, aggregate Greeks, build a P&L grid.

Sign conventions: every position-level number is from the holder's point of
view. ``units`` = +/- quantity x multiplier (100 for options, 1 for stock),
negative for short positions, so a short put has positive delta and positive
theta, exactly as a broker would show it.

Simplifications, stated in the UI and README:
- Options are repriced with Black-Scholes at today's implied vol (plus an
  optional parallel shift); the real smile would move with the stock.
- In the P&L grid, every underlying moves by the same percentage.
- After an option's expiry date it is held at intrinsic value, as if it
  had been exercised or assigned at that price.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from market_hours import ET, days_to_expiry, years_to_expiry
from pricing import bs_price, greeks

POSITION_COLUMNS = ["ticker", "type", "side", "strike", "expiry", "quantity", "entry_price"]
MULTIPLIER = {"call": 100, "put": 100, "stock": 1}
NEAR_STRIKE_PCT = 0.05      # flag short puts within 5% of the strike...
EXPIRING_SOON_DAYS = 7      # ...that expire in under 7 days
MIN_IV = 0.01


# --- Parsing ---------------------------------------------------------------

def empty_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=POSITION_COLUMNS).astype(
        {"strike": float, "quantity": float, "entry_price": float}
    )


def _parse_row(row: dict, today: date) -> dict:
    ticker = str(row.get("ticker") or "").strip().upper()
    if not ticker or ticker == "NAN":
        raise ValueError("missing ticker")

    kind = str(row.get("type") or "").strip().lower()
    if kind not in MULTIPLIER:
        raise ValueError(f"type must be call, put or stock (got {row.get('type')!r})")

    side = str(row.get("side") or "").strip().lower()
    if side not in ("long", "short"):
        raise ValueError(f"side must be long or short (got {row.get('side')!r})")

    quantity = pd.to_numeric(row.get("quantity"), errors="coerce")
    if not quantity > 0:
        raise ValueError("quantity must be a positive number (use side=short for short positions)")

    entry_price = pd.to_numeric(row.get("entry_price"), errors="coerce")
    if not entry_price >= 0:
        raise ValueError("entry_price must be zero or positive")

    strike, expiry = math.nan, None
    if kind != "stock":
        strike = pd.to_numeric(row.get("strike"), errors="coerce")
        if not strike > 0:
            raise ValueError("options need a positive strike")
        parsed = pd.to_datetime(row.get("expiry"), errors="coerce")
        if pd.isna(parsed):
            raise ValueError("options need an expiry date (YYYY-MM-DD)")
        expiry = parsed.date()
        if expiry < today:
            raise ValueError(f"{ticker} {kind} expired on {expiry:%b %d, %Y}")

    return {
        "ticker": ticker, "type": kind, "side": side, "strike": float(strike), "expiry": expiry,
        "quantity": float(quantity), "entry_price": float(entry_price),
    }


def parse_positions(raw: pd.DataFrame, today: date) -> tuple[pd.DataFrame, list[str]]:
    """Validate positions from a CSV or the editor.

    Returns the clean positions and a list of human-readable problems for the
    rows that were skipped, so one bad row never blocks the rest.
    """
    df = raw.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    missing = [c for c in POSITION_COLUMNS if c not in df.columns]
    if missing:
        return empty_positions(), [f"Missing column(s): {', '.join(missing)}. Expected: {', '.join(POSITION_COLUMNS)}."]

    rows, problems = [], []
    for number, row in enumerate(df.to_dict("records"), start=1):
        if all(pd.isna(row.get(c)) or str(row.get(c)).strip() == "" for c in POSITION_COLUMNS):
            continue  # blank line, e.g. a fresh row in the editor
        try:
            rows.append(_parse_row(row, today))
        except ValueError as exc:
            problems.append(f"Row {number}: {exc}")
    positions = pd.DataFrame(rows, columns=POSITION_COLUMNS) if rows else empty_positions()
    return positions, problems


# --- Market inputs ---------------------------------------------------------

def lookup_iv(contracts: pd.DataFrame, strike: float, expiry: date) -> float:
    """IV for a strike/expiry, read off the market's smile.

    Uses the listed expiry closest to ``expiry`` and interpolates linearly in
    strike (flat beyond the last listed strike). NaN if there is no data.
    """
    if contracts.empty:
        return math.nan
    nearest = min(contracts["expiry"].unique(), key=lambda e: abs((e - expiry).days))
    smile = contracts[contracts["expiry"] == nearest].sort_values("strike")
    return float(np.interp(strike, smile["strike"], smile["iv"]))


def position_label(row) -> str:
    side = "Long" if row.side == "long" else "Short"
    qty = f"{row.quantity:g}"
    if row.type == "stock":
        return f"{side} {qty} {row.ticker} shares"
    return f"{side} {qty} {row.ticker} ${row.strike:g} {row.type} {row.expiry:%b %d}"


# --- Valuation -------------------------------------------------------------

def value_positions(
    positions: pd.DataFrame,
    spots: dict[str, float],
    dividend_yields: dict[str, float],
    ivs: list[float],
    r: float,
    now: datetime,
) -> pd.DataFrame:
    """Add current value, P&L, position Greeks, collateral, max loss and flags."""
    df = positions.reset_index(drop=True).copy()
    df["spot"] = df["ticker"].map(spots)
    df["q"] = df["ticker"].map(dividend_yields).fillna(0.0)
    df["iv"] = ivs
    df["sign"] = np.where(df["side"] == "long", 1, -1)
    df["units"] = df["sign"] * df["quantity"] * df["type"].map(MULTIPLIER)
    df["label"] = [position_label(row) for row in df.itertuples(index=False)]

    records = []
    for row in df.itertuples(index=False):
        if row.type == "stock":
            value, g, dte = row.spot, {"delta": 1.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}, None
        else:
            T = years_to_expiry(row.expiry, now)
            value = bs_price(row.spot, row.strike, T, r, row.iv, row.type, row.q)
            g = greeks(row.spot, row.strike, T, r, row.iv, row.type, row.q)
            dte = days_to_expiry(row.expiry, now)
        records.append({
            "dte": dte,
            "value": value,
            "market_value": value * row.units,
            "pnl": (value - row.entry_price) * row.units,
            **{name: g[name] * row.units for name in ("delta", "gamma", "theta", "vega")},
        })
    df = pd.concat([df, pd.DataFrame(records, index=df.index)], axis=1)

    df["covered"] = _covered_calls(df)
    df["collateral"] = np.where(
        (df["type"] == "put") & (df["side"] == "short"), df["strike"] * 100 * df["quantity"], 0.0
    )
    df["max_loss"] = [_max_loss(row) for row in df.itertuples(index=False)]
    df["flag"] = [_flag(row) for row in df.itertuples(index=False)]
    return df


def _covered_calls(df: pd.DataFrame) -> pd.Series:
    """Short calls backed by enough long shares of the same ticker (100 per contract)."""
    shares = df[(df["type"] == "stock") & (df["side"] == "long")].groupby("ticker")["quantity"].sum()
    short_calls = (df["type"] == "call") & (df["side"] == "short")
    contracts_short = df[short_calls].groupby("ticker")["quantity"].sum()
    coverable = (shares.reindex(contracts_short.index).fillna(0) >= contracts_short * 100)
    return short_calls & df["ticker"].map(coverable).eq(True)


def _max_loss(row) -> float:
    """Worst case at expiry for this position alone (inf = unlimited)."""
    if row.side == "long":
        return row.entry_price * abs(row.units)  # premium paid, or stock to zero
    if row.type == "put":
        return (row.strike - row.entry_price) * abs(row.units)  # assigned, stock goes to zero
    if row.type == "call" and row.covered:
        return 0.0  # the downside sits in the stock leg; upside is capped, not lost
    return math.inf  # uncovered short call or short stock


def _flag(row) -> str:
    if row.type == "call" and row.side == "short" and not row.covered:
        return "Uncovered short call: unlimited risk"
    if not (row.type == "put" and row.side == "short"):
        return ""
    in_the_money = row.spot < row.strike
    near_strike = row.spot <= row.strike * (1 + NEAR_STRIKE_PCT)
    dte = int(row.dte)
    soon = dte < EXPIRING_SOON_DAYS
    days = f"{dte} day{'s' if dte != 1 else ''}"
    if in_the_money and soon:
        return f"In the money, expires in {days}: assignment likely"
    if in_the_money:
        return "In the money"
    if near_strike and soon:
        return f"Within {NEAR_STRIKE_PCT:.0%} of strike, expires in {days}"
    return ""


def portfolio_totals(valued: pd.DataFrame) -> dict:
    return {
        "market_value": valued["market_value"].sum(),
        "pnl": valued["pnl"].sum(),
        "delta": valued["delta"].sum(),
        "gamma": valued["gamma"].sum(),
        "theta": valued["theta"].sum(),
        "vega": valued["vega"].sum(),
        "collateral": valued["collateral"].sum(),
        # Summing per-position worst cases overstates the loss on spreads, so
        # this is a conservative bound, not an exact portfolio max loss.
        "max_loss": valued["max_loss"].sum(),
    }


# --- Scenario grid ---------------------------------------------------------

def grid_dates(valued: pd.DataFrame, today: date, max_columns: int = 45) -> list[date]:
    """Today through the last option expiry; every expiry date is always a column."""
    expiries = sorted({e for e in valued["expiry"] if e is not None})
    last = expiries[-1] if expiries else today + timedelta(days=30)
    days = [today + timedelta(days=i) for i in range((last - today).days + 1)]
    if len(days) > max_columns:
        picks = np.unique(np.linspace(0, len(days) - 1, max_columns).round().astype(int))
        days = sorted({days[i] for i in picks} | set(expiries))
    return days


def pnl_grid(
    valued: pd.DataFrame,
    r: float,
    now: datetime,
    moves: np.ndarray | None = None,
    iv_shift: float = 0.0,
) -> pd.DataFrame:
    """Portfolio P&L for each underlying % move (rows) on each future date (columns).

    Today's column uses the current time; later dates are valued at the 4 pm
    close. ``iv_shift`` is in vol points as a decimal (0.05 = +5 pts).
    """
    moves = np.round(np.linspace(-0.20, 0.20, 41), 4) if moves is None else np.asarray(moves)
    today = now.astimezone(ET).date()
    dates = grid_dates(valued, today)
    moments = [now] + [datetime.combine(d, time(16, 0), tzinfo=ET) for d in dates[1:]]

    total = np.zeros((len(moves), len(dates)))
    for row in valued.itertuples(index=False):
        spot_scenarios = row.spot * (1 + moves)[:, None]
        if row.type == "stock":
            value = np.broadcast_to(spot_scenarios, total.shape)
        else:
            T = np.array([years_to_expiry(row.expiry, m) for m in moments])[None, :]
            vol = max(row.iv + iv_shift, MIN_IV)
            value = bs_price(spot_scenarios, row.strike, T, r, vol, row.type, row.q)
        total += (value - row.entry_price) * row.units
    return pd.DataFrame(total, index=moves, columns=dates)
