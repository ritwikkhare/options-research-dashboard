"""Tab 1: implied volatility surface, smile and term structure."""

from __future__ import annotations

import streamlit as st

from analytics.surface import SIDES, atm_term_structure, select_side, smile
from ui import charts
from ui.data import Snapshot, cached_surface
from ui.sidebar import Settings
from ui.theme import CHART_CONFIG


def render(snapshot: Snapshot, settings: Settings) -> None:
    after_hours = not settings.market_open
    surface = cached_surface(
        snapshot.chain, snapshot.spot, settings.risk_free_rate, snapshot.dividend_yield,
        snapshot.fetched_at, after_hours,
    )

    side = st.radio("Contracts", list(SIDES), format_func=SIDES.get, horizontal=True, key="surface_side")
    contracts = select_side(surface, side)
    if contracts.empty:
        st.info(
            f"No liquid {snapshot.ticker} contracts with a usable price in this range. "
            "Try a wider expiration range, or check back during market hours."
        )
        return

    term = atm_term_structure(contracts, snapshot.spot)
    computed_share = (contracts["iv_source"] == "computed").mean()

    spot_col, atm_col, earnings_col, count_col = st.columns(4)
    spot_col.metric("Spot", f"${snapshot.spot:,.2f}")
    atm_col.metric(
        "Front ATM IV",
        f"{term['atm_iv'].iloc[0]:.1%}" if not term.empty else "n/a",
        help="At-the-money implied vol of the nearest expiry in range, interpolated to spot.",
    )
    earnings_col.metric(
        "Next earnings",
        f"{snapshot.earnings_date:%b %d}" if snapshot.earnings_date else "n/a",
        help=f"{snapshot.earnings_date:%A, %B %d, %Y}" if snapshot.earnings_date else
        "The data source has no upcoming earnings date for this ticker.",
    )
    count_col.metric(
        "Contracts plotted",
        f"{len(contracts):,}",
        help=f"{computed_share:.0%} use IV we solved ourselves with Black-Scholes; the rest use "
        "the provider's IV. Illiquid quotes are skipped.",
    )

    if after_hours:
        st.caption(
            ":orange[Market closed:] the free feed shows no live bid/ask, so IV is solved from each "
            "contract's last trade price (if it traded in the last 5 days). Expect a noisier surface "
            "until the open."
        )

    st.plotly_chart(
        charts.iv_heatmap(contracts, snapshot.spot, snapshot.earnings_date,
                          f"{snapshot.ticker} implied volatility surface · {SIDES[side]}"),
        config=CHART_CONFIG,
        theme=None,
        width="stretch",
    )

    smile_col, term_col = st.columns(2)
    with smile_col:
        expiries = sorted(contracts["expiry"].unique())
        expiry = st.selectbox(
            "Smile expiry",
            expiries,
            format_func=lambda e: f"{e:%b %d, %Y} ({(contracts.loc[contracts['expiry'] == e, 'dte'].iloc[0])} DTE)",
        )
        st.plotly_chart(charts.smile_chart(smile(contracts, expiry), snapshot.spot, expiry),
                        config=CHART_CONFIG, theme=None, width="stretch")
    with term_col:
        # Keep the two charts level: a spacer the height of the selectbox.
        st.markdown("<div style='height: 4.6rem'></div>", unsafe_allow_html=True)
        if term.empty:
            st.info("Not enough strikes around spot to interpolate ATM IV.")
        else:
            st.plotly_chart(charts.term_structure_chart(term, snapshot.earnings_date),
                            config=CHART_CONFIG, theme=None, width="stretch")

    with st.expander("How to read this view"):
        st.markdown(
            """
- **Surface:** each cell is the market's implied volatility for one strike and expiry. Brighter = more
  expensive options. The *OTM only* view uses puts below spot and calls above, because out-of-the-money
  options are the most liquid, so their prices (and IVs) are the most reliable.
- **Smile / skew:** for most stocks, IV rises for lower strikes, because downside protection is in
  demand. A steep left side means puts are rich.
- **Term structure:** ATM IV by expiry. Normally it slopes gently upward; a bump at the first expiry
  after earnings is the market pricing the earnings move.
- **Open circles:** the provider's IV was missing or unrealistic, so we solved for IV from the bid/ask
  mid with Black-Scholes.
            """
        )
