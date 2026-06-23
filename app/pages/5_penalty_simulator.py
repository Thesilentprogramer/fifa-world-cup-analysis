"""Phase 4 — Penalty Shootout Simulator.

Features:
• Team A / Team B selectors from the match predictor's team list
• Goalkeeper selector (named presets with skill modifiers)
• 5-taker roster with skill presets per player position
• 10,000-simulation Monte Carlo win probabilities (Plotly gauge + bar)
• Interactive kick-by-kick mode with premium grid scoreboard
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.penalty_engine import (
    KEEPER_PRESETS,
    TAKER_PRESETS,
    BASE_CONVERSION,
    simulate_shootout,
    simulate_one_shootout,
)
from src.match_predictor import get_predictor
from app.theme import inject_app_styles

inject_app_styles()

st.title("🥅 Penalty Shootout Simulator")
st.caption(
    "Monte Carlo simulation · named goalkeeper presets · pressure-adjusted conversion rates · "
    "interactive kick-by-kick mode"
)


@st.cache_resource
def load_predictor():
    return get_predictor()


try:
    predictor = load_predictor()
    teams = sorted(predictor.teams)
except Exception:
    teams = [
        "Argentina", "Brazil", "France", "England", "Germany", "Spain",
        "Portugal", "Netherlands", "Italy", "Croatia", "Morocco", "Japan",
        "United States", "Mexico", "Senegal", "Australia",
    ]

# ---------------------------------------------------------------------------
# Sidebar — team & goalkeeper setup
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🏟️ Match Setup")

    team_a = st.selectbox(
        "Team A",
        teams,
        index=teams.index("Argentina") if "Argentina" in teams else 0,
        key="ps_team_a",
    )
    team_b = st.selectbox(
        "Team B",
        [t for t in teams if t != team_a],
        index=([t for t in teams if t != team_a].index("France")
               if "France" in teams and "France" != team_a else 0),
        key="ps_team_b",
    )

    st.divider()
    st.header("🧤 Goalkeepers")
    keeper_options = list(KEEPER_PRESETS.keys())

    keeper_a_name = st.selectbox(f"{team_a} keeper", keeper_options, index=1, key="ka")
    keeper_b_name = st.selectbox(f"{team_b} keeper", keeper_options, index=0, key="kb")

    n_sims = st.select_slider(
        "Simulations",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
    )


# ---------------------------------------------------------------------------
# Taker skill inputs (main area)
# ---------------------------------------------------------------------------

st.subheader("🎯 Penalty Takers")

taker_options = list(TAKER_PRESETS.keys())
taker_defaults = {
    1: "Expert (specialist)",
    2: "Reliable",
    3: "Reliable",
    4: "Average player",
    5: "Average player",
}

col_a, col_b = st.columns(2)

takers_a: list[float] = []
with col_a:
    st.markdown(f"**{team_a} takers**")
    for i in range(1, 6):
        choice = st.selectbox(
            f"Taker {i}",
            taker_options,
            index=taker_options.index(taker_defaults.get(i, "Average player")),
            key=f"ta_{i}",
        )
        takers_a.append(TAKER_PRESETS[choice]["skill"])

takers_b: list[float] = []
with col_b:
    st.markdown(f"**{team_b} takers**")
    for i in range(1, 6):
        choice = st.selectbox(
            f"Taker {i}",
            taker_options,
            index=taker_options.index(taker_defaults.get(i, "Average player")),
            key=f"tb_{i}",
        )
        takers_b.append(TAKER_PRESETS[choice]["skill"])

keeper_a_save = KEEPER_PRESETS[keeper_a_name]["save_modifier"]
keeper_b_save = KEEPER_PRESETS[keeper_b_name]["save_modifier"]

# ---------------------------------------------------------------------------
# Run Monte Carlo
# ---------------------------------------------------------------------------

run_btn = st.button("▶️  Run Simulation", type="primary", use_container_width=True)

if run_btn or "ps_result" in st.session_state:
    if run_btn:
        with st.spinner(f"Running {n_sims:,} shootout simulations…"):
            result = simulate_shootout(
                takers_a, takers_b,
                keeper_a_save_modifier=keeper_a_save,
                keeper_b_save_modifier=keeper_b_save,
                n_simulations=n_sims,
                seed=None,  # fresh random each run
            )
        st.session_state["ps_result"] = result
        st.session_state["ps_team_a_name"] = team_a
        st.session_state["ps_team_b_name"] = team_b
    else:
        result = st.session_state["ps_result"]
        team_a = st.session_state.get("ps_team_a_name", team_a)
        team_b = st.session_state.get("ps_team_b_name", team_b)

    win_a = result["win_prob_a"]
    win_b = result["win_prob_b"]
    avg_rounds = result["avg_rounds"]

    st.divider()
    st.subheader("📊 Simulation Results")

    # Gauge for Team A
    fig_gauge = go.Figure()
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=win_a * 100,
        domain={"x": [0, 0.48], "y": [0, 1]},
        title={"text": f"{team_a}<br><sub>Win probability</sub>", "font": {"size": 16}},
        delta={"reference": 50, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#6C63FF"},
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.15)"},
                {"range": [40, 60], "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 100], "color": "rgba(16,185,129,0.15)"},
            ],
            "threshold": {"line": {"color": "#6C63FF", "width": 3}, "value": 50},
        },
        number={"suffix": "%", "font": {"size": 32}},
    ))
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=win_b * 100,
        domain={"x": [0.52, 1.0], "y": [0, 1]},
        title={"text": f"{team_b}<br><sub>Win probability</sub>", "font": {"size": 16}},
        delta={"reference": 50, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#F59E0B"},
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.15)"},
                {"range": [40, 60], "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 100], "color": "rgba(16,185,129,0.15)"},
            ],
            "threshold": {"line": {"color": "#F59E0B", "width": 3}, "value": 50},
        },
        number={"suffix": "%", "font": {"size": 32}},
    ))
    fig_gauge.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_gauge, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{team_a} wins", f"{win_a:.1%}")
    c2.metric(f"{team_b} wins", f"{win_b:.1%}")
    c3.metric("Avg rounds", f"{avg_rounds:.1f}")

    # Keeper info
    kinfo_a = KEEPER_PRESETS[keeper_a_name]
    kinfo_b = KEEPER_PRESETS[keeper_b_name]
    ka_eff = BASE_CONVERSION - kinfo_a["save_modifier"]
    kb_eff = BASE_CONVERSION - kinfo_b["save_modifier"]
    st.caption(
        f"Goalkeeper effect · {team_a} ({kinfo_a['flag']} {keeper_a_name}): "
        f"opp conversion reduced to {ka_eff:.1%}  |  "
        f"{team_b} ({kinfo_b['flag']} {keeper_b_name}): "
        f"opp conversion reduced to {kb_eff:.1%}"
    )

    # ---------------------------------------------------------------------------
    # Interactive kick-by-kick simulation
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("⚡ Interactive Kick-by-Kick")
    st.caption("Step through the sample shootout returned from the simulation.")

    sample_log = result["sample_log"]

    if "ps_step" not in st.session_state:
        st.session_state["ps_step"] = 0

    col_prev, col_next, col_reset = st.columns([1, 1, 1])
    with col_prev:
        if st.button("⬅️ Previous kick", disabled=st.session_state["ps_step"] == 0):
            st.session_state["ps_step"] = max(0, st.session_state["ps_step"] - 1)
    with col_next:
        if st.button("➡️ Next kick", disabled=st.session_state["ps_step"] >= len(sample_log)):
            st.session_state["ps_step"] = min(len(sample_log), st.session_state["ps_step"] + 1)
    with col_reset:
        if st.button("🔄 Reset"):
            st.session_state["ps_step"] = 0

    current_step = st.session_state["ps_step"]
    kicks_so_far = sample_log[:current_step]

    # Build grid display
    # Rows: round 1-5+ | Cols: Team A kick, Team B kick
    SCORE_ICON = "🟢"
    MISS_ICON = "🔴"
    PENDING_ICON = "⚪"

    a_kicks = [k for k in kicks_so_far if k["team"] == "a"]
    b_kicks = [k for k in kicks_so_far if k["team"] == "b"]
    all_rounds = sorted(set(k["round"] for k in sample_log))

    score_a = kicks_so_far[-1]["score_a"] if kicks_so_far else 0
    score_b = kicks_so_far[-1]["score_b"] if kicks_so_far else 0

    st.markdown(
        f"""
        <div style="text-align:center; font-size:2rem; margin: 1rem 0; font-family:'Syne',sans-serif; font-weight:800; letter-spacing:0.05em;">
            {team_a}  <span style="color:#6C63FF">{score_a}</span>
            &nbsp;—&nbsp;
            <span style="color:#F59E0B">{score_b}</span>  {team_b}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Grid
    grid_rows = []
    for r in sorted(set(k["round"] for k in sample_log)):
        a_kick_this = next((k for k in a_kicks if k["round"] == r), None)
        b_kick_this = next((k for k in b_kicks if k["round"] == r), None)
        a_full = next((k for k in sample_log if k["round"] == r and k["team"] == "a"), None)
        b_full = next((k for k in sample_log if k["round"] == r and k["team"] == "b"), None)

        a_icon = (SCORE_ICON if a_kick_this and a_kick_this["scored"] else MISS_ICON) if a_kick_this else PENDING_ICON
        b_icon = (SCORE_ICON if b_kick_this and b_kick_this["scored"] else MISS_ICON) if b_kick_this else PENDING_ICON

        a_prob = f"{a_full['prob']:.0%}" if a_full else ""
        b_prob = f"{b_full['prob']:.0%}" if b_full else ""

        grid_rows.append(f"| Round {r} | {a_icon} {a_prob} | {b_icon} {b_prob} |")

    grid_header = f"| Round | {team_a} | {team_b} |\n|-------|-------|-------|\n"
    st.markdown(grid_header + "\n".join(grid_rows))

    # Progress info
    if current_step == 0:
        st.info("Press **Next kick ➡️** to begin the shootout.")
    elif current_step >= len(sample_log):
        final_a = sample_log[-1]["score_a"]
        final_b = sample_log[-1]["score_b"]
        winner = team_a if final_a > final_b else team_b
        st.success(f"🏆 **{winner}** wins the shootout! Final: {final_a}–{final_b}")
    else:
        last_kick = kicks_so_far[-1]
        kicker_team = team_a if last_kick["team"] == "a" else team_b
        result_txt = "⚽ **SCORED**" if last_kick["scored"] else "❌ **MISSED**"
        st.info(f"Round {last_kick['round']} · {kicker_team}: {result_txt} (conversion prob was {last_kick['prob']:.0%})")

else:
    st.info("Configure the setup above and click **▶️ Run Simulation** to begin.")
