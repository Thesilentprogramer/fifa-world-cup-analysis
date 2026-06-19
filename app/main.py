"""FIFA World Cup Match Predictor — Streamlit entry point."""

import streamlit as st

st.set_page_config(
    page_title="FIFA World Cup Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FIFA World Cup Match Predictor")
st.markdown(
    """
    Predict international football match outcomes with calibrated Win / Draw / Loss probabilities.

    **Data sources:** StatsBomb Open Data · Transfermarkt Datasets

    Use the sidebar to navigate to **Match Predictor**.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", "XGBoost + Isotonic Calibration")
with col2:
    st.metric("Validation", "2018 World Cup")
with col3:
    st.metric("Test Set", "2022 World Cup (held-out)")

st.info(
    "Probabilities are estimates based on historical data. "
    "Navigate to **Match Predictor** in the sidebar to compare two national teams."
)
