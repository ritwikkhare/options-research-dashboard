"""Options research dashboard: Streamlit entry point.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import logging
from datetime import timedelta

import streamlit as st

from market_hours import is_market_open, now_et
from providers import DataProviderError
from ui import tab_risk, tab_screener, tab_surface
from ui.data import QUOTE_TTL, load_snapshot
from ui.sidebar import render_sidebar, show_last_updated

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Options Research Dashboard", page_icon="📈", layout="wide")


@st.fragment(run_every=timedelta(seconds=QUOTE_TTL) if is_market_open() else None)
def auto_refresh() -> None:
    """While the market is open, rerun the whole app once the cached quotes expire.

    The fragment ticks on its own timer; a full rerun resets ``loaded_at``,
    so this never loops.
    """
    loaded_at = st.session_state.get("loaded_at")
    if loaded_at and (now_et() - loaded_at).total_seconds() >= QUOTE_TTL:
        st.rerun()


def show_safely(render, *args) -> None:
    """Run one tab, turning any failure into a friendly message instead of a stack trace."""
    try:
        render(*args)
    except DataProviderError as exc:
        st.info(str(exc))
    except Exception:
        logger.exception("Unexpected error rendering %s", getattr(render, "__module__", render))
        st.error("Something went wrong building this view. Try refreshing, or pick another ticker.")


def main() -> None:
    st.session_state.loaded_at = now_et()
    settings, last_updated_slot = render_sidebar()

    st.title(f"{settings.ticker} options research")

    snapshot, load_error = None, None
    with st.spinner(f"Loading {settings.ticker} option chains..."):
        try:
            snapshot = load_snapshot(settings.ticker, settings.min_dte, settings.max_dte)
        except DataProviderError as exc:
            load_error = str(exc)
        except Exception:
            logger.exception("Unexpected error loading %s", settings.ticker)
            load_error = f"Couldn't load data for {settings.ticker} right now. Try refreshing."
    show_last_updated(last_updated_slot, snapshot.fetched_at if snapshot else None)

    surface_tab, screener_tab, risk_tab = st.tabs(["Volatility surface", "Put screener", "Risk"])
    with surface_tab:
        if snapshot is None:
            st.info(load_error)
        else:
            show_safely(tab_surface.render, snapshot, settings)
    with screener_tab:
        show_safely(tab_screener.render, settings)
    with risk_tab:
        show_safely(tab_risk.render, settings)

    auto_refresh()


main()
