"""Match Outcome Predictor page."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.match_predictor import get_predictor

st.title("Match Outcome Predictor")
st.caption("Calibrated Win / Draw / Loss probabilities with SHAP explanations")

@st.cache_resource
def load_predictor():
    return get_predictor()

try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(f"Model not found. Run `python scripts/train_model.py` first.\n\n{e}")
    st.stop()

teams = sorted(predictor.teams)

with st.sidebar:
    st.header("Match Setup")
    team_a = st.selectbox("Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    team_b = st.selectbox(
        "Team B",
        [t for t in teams if t != team_a],
        index=([t for t in teams if t != team_a].index("Argentina")
                if "Argentina" in teams and "Argentina" != team_a else 0),
    )
    stage = st.selectbox(
        "Tournament Stage",
        ["group", "round_of_16", "quarter_final", "semi_final", "final"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    is_home = st.radio("Team A plays at", ["Home", "Neutral/Away"], index=0)
    predict_btn = st.button("Predict", type="primary", use_container_width=True)

if predict_btn or True:
    result = predictor.predict(
        team_a, team_b, stage=stage, is_home=1 if is_home == "Home" else 0
    )
    probs = result["probabilities"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"{team_a} vs {team_b}")
        outcome = result["predicted_outcome"]
        if outcome == "Win":
            st.success(f"Predicted: **{team_a} Win** ({result['confidence']:.0%})")
        elif outcome == "Loss":
            st.success(f"Predicted: **{team_b} Win** ({result['confidence']:.0%})")
        else:
            st.warning(f"Predicted: **Draw** ({result['confidence']:.0%})")

        if result.get("sparse_data_warning"):
            st.warning("Limited historical data for one or both teams — predictions may be less reliable.")

        labels = [f"{team_a} Win", "Draw", f"{team_b} Win"]
        values = [probs["win"], probs["draw"], probs["loss"]]
        colors = ["#2E7D32", "#F9A825", "#C62828"]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textposition="outside",
            pull=[0.05 if v == max(values) else 0 for v in values],
        )])
        fig.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=400,
            annotations=[dict(text="Outcome", x=0.5, y=0.5, font_size=16, showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Why this prediction?")
        with st.expander(f"Why did the model favor this outcome?", expanded=True):
            st.markdown(result["narrative"])

        shap_data = result["shap"]
        if shap_data:
            features = [d["label"] for d in shap_data][::-1]
            values = [d["shap_value"] for d in shap_data][::-1]
            colors_bar = ["#2E7D32" if v > 0 else "#C62828" for v in values]

            fig_shap = go.Figure(go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker_color=colors_bar,
            ))
            fig_shap.update_layout(
                title="Top 5 SHAP Features",
                xaxis_title="SHAP value (impact on prediction)",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{team_a} Win", f"{probs['win']:.1%}")
    c2.metric("Draw", f"{probs['draw']:.1%}")
    c3.metric(f"{team_b} Win", f"{probs['loss']:.1%}")

st.caption(
    "Model trained on StatsBomb international match data. "
    "Probabilities are calibrated estimates, not guarantees."
)
