"""xG Engine — shot model + Monte Carlo match simulation (Phase 2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.theme import inject_app_styles
from app.charts.model_charts import (
    fig_xg_calibration,
    fig_xg_distance_curve,
    load_xg_metrics,
)

inject_app_styles()

st.title("xG Engine")
st.caption("Shot-level expected goals model + Monte Carlo scoreline simulation")

try:
    from src.statsbomb_shots import load_statsbomb_shots
    from src.xg_engine import load_xg_model, simulate_match, team_xg_rates_from_history
    from src.match_predictor import get_predictor

    predictor = get_predictor()
    teams = sorted(predictor.teams)
    shots = load_statsbomb_shots()
    model = load_xg_model()
except (FileNotFoundError, ImportError, OSError) as e:
    st.error(f"Missing model or data. Run training scripts first.\n\n{e}")
    st.stop()

# --- Model Performance showcase ---
st.markdown('<div class="model-perf-header"><h3>Model Performance</h3></div>', unsafe_allow_html=True)
xg_m = load_xg_metrics()
xc1, xc2, xc3, xc4 = st.columns(4)
xc1.metric("ROC-AUC", f"{xg_m.get('roc_auc', 0):.3f}" if xg_m else "—")
xc2.metric("Shots trained", f"{xg_m.get('n_shots', 0):,}" if xg_m else "—")
xc3.metric("Mean pred xG", f"{xg_m.get('mean_predicted_xg', 0):.3f}" if xg_m else "—")
xc4.metric("StatsBomb xG", f"{xg_m.get('mean_statsbomb_xg', 0):.3f}" if xg_m else "—")

cal_c1, cal_c2 = st.columns(2)
with cal_c1:
    st.plotly_chart(fig_xg_calibration(shots, model), width="stretch")
with cal_c2:
    st.plotly_chart(fig_xg_distance_curve(shots, model), width="stretch")

st.divider()

with st.sidebar:
    team_a = st.selectbox("Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    team_b = st.selectbox(
        "Team B",
        [t for t in teams if t != team_a],
        index=([t for t in teams if t != team_a].index("France") if "France" in teams else 0),
    )
    n_sims = st.slider("Simulations", 1000, 20000, 10000, step=1000)
    run = st.button("Simulate Match", type="primary")

if run or True:
    result = simulate_match(team_a, team_b, n_simulations=n_sims, shots=shots, model=model)
    ra = team_xg_rates_from_history(team_a, shots, model)
    rb = team_xg_rates_from_history(team_b, shots, model)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{team_a} xG", f"{result['expected_xg_a']:.2f}")
    c2.metric("Draw", f"{result['prob_draw']:.0%}")
    c3.metric(f"{team_b} xG", f"{result['expected_xg_b']:.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric(f"{team_a} win", f"{result['prob_win_a']:.0%}")
    c5.metric("Most likely score", result["top_scorelines"][0]["score_a"] if result["top_scorelines"] else "-")
    c6.metric(f"{team_b} win", f"{result['prob_win_b']:.0%}")

    st.subheader("Scoreline probabilities")
    sc = pd.DataFrame(result["top_scorelines"])
    if not sc.empty:
        sc["label"] = sc.apply(lambda r: f"{int(r['score_a'])}-{int(r['score_b'])}", axis=1)
        fig = px.bar(sc, x="label", y="probability", labels={"probability": "Probability", "label": "Scoreline"})
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Shot data coverage")
    st.write(
        f"{team_a}: {ra['n_matches']} matches with shot data · "
        f"{team_b}: {rb['n_matches']} matches with shot data · "
        f"Total shots in corpus: {len(shots):,}"
    )

    # Pitch scatter for recent team shots
    try:
        from mplsoccer import VerticalPitch
        import matplotlib.pyplot as plt

        team_shots = shots[shots["team"] == team_a].tail(200)
        if not team_shots.empty:
            st.subheader(f"{team_a} — recent shot locations")
            pitch = VerticalPitch(pitch_type="statsbomb", line_zorder=2)
            fig_mpl, ax = pitch.draw(figsize=(6, 8))
            goals = team_shots["is_goal"] == 1
            pitch.scatter(
                team_shots.loc[~goals, "x"], team_shots.loc[~goals, "y"],
                ax=ax, s=30, c="steelblue", alpha=0.5, label="Shot",
            )
            pitch.scatter(
                team_shots.loc[goals, "x"], team_shots.loc[goals, "y"],
                ax=ax, s=60, c="gold", edgecolors="black", label="Goal",
            )
            ax.legend(loc="upper right")
            st.pyplot(fig_mpl)
    except ImportError:
        st.info("Install mplsoccer for pitch visualization: pip install mplsoccer")
