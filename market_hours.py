"""US equity market clock: trading hours and time-to-expiry.

All times are US/Eastern, the exchange's time zone, regardless of where the
app is hosted (Streamlit Community Cloud servers run on UTC).
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
SECONDS_PER_YEAR = 365 * 24 * 60 * 60

# Full-day NYSE closures. Half days (e.g. day after Thanksgiving) are treated
# as normal days; that only affects the auto-refresh schedule. Extend yearly.
NYSE_HOLIDAYS = {
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}


def now_et() -> datetime:
    return datetime.now(ET)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in NYSE_HOLIDAYS


def is_market_open(now: datetime | None = None) -> bool:
    """True during regular trading hours (9:30-16:00 ET on trading days)."""
    now = (now or now_et()).astimezone(ET)
    return is_trading_day(now.date()) and MARKET_OPEN <= now.time() < MARKET_CLOSE


def days_to_expiry(expiry: date, now: datetime | None = None) -> int:
    """Calendar days until expiry (DTE), as quoted by brokers."""
    now = (now or now_et()).astimezone(ET)
    return (expiry - now.date()).days


def years_to_expiry(expiry: date, now: datetime | None = None) -> float:
    """Time to expiry in years, measured to the 4:00 pm ET close on expiry day.

    Using the exact close instead of whole days matters for short-dated
    options: a 7-DTE option checked at 3 pm has noticeably less time left
    than one checked at 9:30 am. Returns 0 once the option has expired.
    """
    now = (now or now_et()).astimezone(ET)
    expiry_close = datetime.combine(expiry, MARKET_CLOSE, tzinfo=ET)
    return max((expiry_close - now).total_seconds(), 0.0) / SECONDS_PER_YEAR
