"""FIFA World Cup Match Predictor — splash landing."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))

from app.theme import inject_splash_styles, render_splash_background

st.set_page_config(
    page_title="FIFA World Cup Match Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

inject_splash_styles()
render_splash_background(height=800)

st.markdown(
    """
    <div class="splash-content">
      <div class="splash-title">FIFA World Cup<br>Match Analysis Dashboard</div>
      <div class="splash-subtitle">
        Calibrated W/D/L predictions · xG simulation · SHAP explainability · WC 2026 fixtures
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_, col_btn, _ = st.columns([1, 1.2, 1])
with col_btn:
    st.markdown('<div class="splash-btn-wrap">', unsafe_allow_html=True)
    if st.button("Enter Dashboard", type="primary", use_container_width=True):
        st.switch_page("pages/1_match_predictor.py")
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("About this project"):
    st.markdown(
        """
        **Match Outcome Model** — XGBoost + isotonic calibration on men's internationals since 2000.
        Validated on 2018 World Cup, tested on held-out 2022 World Cup.

        **Data sources:** martj42 internationals · StatsBomb · Transfermarkt API · Polymarket · API-Football

        | Metric | Value |
        |--------|-------|
        | Validation (2018 WC) | ~62.5% accuracy |
        | Test (2022 WC) | ~56.3% accuracy |
        | xG shot model AUC | ~0.81 |

        Use the sidebar after entering the dashboard: **WC 2026 Dashboard** · **xG Engine**
        """
    )

st.caption("Probabilities are estimates based on historical data — not betting advice.")
