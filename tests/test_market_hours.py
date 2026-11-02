from datetime import date, datetime

import pytest

from market_hours import ET, SECONDS_PER_YEAR, days_to_expiry, is_market_open, years_to_expiry


def et(*args) -> datetime:
    return datetime(*args, tzinfo=ET)


@pytest.mark.parametrize(
    "moment, expected",
    [
        (et(2026, 11, 2, 10, 0), True),    # Monday mid-morning
        (et(2026, 11, 2, 9, 29), False),   # before the open
        (et(2026, 11, 2, 16, 0), False),   # at the close
        (et(2026, 11, 7, 12, 0), False),   # Saturday
        (et(2026, 11, 26, 12, 0), False),  # Thanksgiving
    ],
)
def test_is_market_open(moment, expected):
    assert is_market_open(moment) is expected


def test_market_open_uses_eastern_time_from_utc_server_clock():
    # 14:45 UTC on Nov 2 is 9:45 am ET (EST, UTC-5): market is open.
    utc_moment = datetime.fromisoformat("2026-11-02T14:45:00+00:00")
    assert is_market_open(utc_moment)


def test_days_to_expiry_counts_calendar_days():
    assert days_to_expiry(date(2026, 11, 20), et(2026, 11, 2, 15, 0)) == 18


def test_years_to_expiry_measures_to_the_4pm_close():
    # Monday 4 pm to Friday 4 pm the same week is exactly 4 days.
    assert years_to_expiry(date(2026, 11, 6), et(2026, 11, 2, 16, 0)) == pytest.approx(
        4 * 86400 / SECONDS_PER_YEAR
    )


def test_years_to_expiry_is_zero_after_expiry():
    assert years_to_expiry(date(2026, 11, 6), et(2026, 11, 6, 16, 30)) == 0.0
