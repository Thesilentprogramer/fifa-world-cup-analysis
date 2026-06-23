"""Phase 5 & 6 — Knockout Bracket Simulator.

Features:
- Select presets: 2022 actual Round of 16, 2026 Elo favorites, or Custom Selection.
- Simulate round-by-round (handling draws with the penalty shootout simulator).
- Simulate entire tournament to the champion.
- Beautiful visual styled tree bracket using side-by-side flexbox columns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import random

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.match_predictor import get_predictor
from src.penalty_engine import simulate_one_shootout
from app.theme import inject_app_styles

inject_app_styles()

st.title("🏆 Knockout Bracket Simulator")
st.caption(
    "Interactive tournament simulator · regular time goal simulation · "
    "penalty shootout tiebreakers · visual tree bracket view"
)

# Load predictor
try:
    predictor = get_predictor()
except FileNotFoundError as e:
    st.error(f"Model not found. Run `python scripts/train_model.py` first.\n\n{e}")
    st.stop()

teams_list_all = sorted(predictor.teams)

# Load flag ISO mapping from wc2026_nations.json
NATIONS_JSON = PROJECT_ROOT / "app" / "assets" / "wc2026_nations.json"
name_to_iso = {}
if NATIONS_JSON.exists():
    try:
        nations_data = json.loads(NATIONS_JSON.read_text(encoding="utf-8"))
        for item in nations_data:
            name_to_iso[item["name"]] = item["iso"]
    except Exception:
        pass

# Fallback / historical team ISOs
FALLBACK_ISO = {
    "Argentina": "ar", "Brazil": "br", "France": "fr", "England": "gb-eng",
    "Germany": "de", "Spain": "es", "Portugal": "pt", "Netherlands": "nl",
    "Belgium": "be", "Croatia": "hr", "Switzerland": "ch", "Morocco": "ma",
    "Japan": "jp", "South Korea": "kr", "United States": "us", "Mexico": "mx",
    "Senegal": "sn", "Australia": "au", "Poland": "pl", "Denmark": "dk",
    "Tunisia": "tn", "Costa Rica": "cr", "Serbia": "rs", "Cameroon": "cm",
    "Ghana": "gh", "Uruguay": "uy", "Ecuador": "ec", "Canada": "ca",
    "Saudi Arabia": "sa", "Iran": "ir", "Qatar": "qa", "Wales": "gb-wls",
    "Czechia": "cz", "Turkey": "tr", "Norway": "no", "Sweden": "se",
    "Austria": "at", "Italy": "it"
}

def get_flag_html(team_name: str) -> str:
    iso = name_to_iso.get(team_name) or FALLBACK_ISO.get(team_name)
    if iso:
        return f'<img class="flag-img" src="https://flagcdn.com/w40/{iso.lower()}.png" alt="{team_name}">'
    return '<span style="margin-right:8px;">🏳️</span>'


# ---------------------------------------------------------------------------
# Bracket state management
# ---------------------------------------------------------------------------

if "bracket_round_of_16" not in st.session_state:
    st.session_state["bracket_round_of_16"] = []
if "bracket_quarter_finals" not in st.session_state:
    st.session_state["bracket_quarter_finals"] = []
if "bracket_semi_finals" not in st.session_state:
    st.session_state["bracket_semi_finals"] = []
if "bracket_final" not in st.session_state:
    st.session_state["bracket_final"] = []
if "bracket_champion" not in st.session_state:
    st.session_state["bracket_champion"] = None
if "bracket_current_round" not in st.session_state:
    st.session_state["bracket_current_round"] = "uninitialized"


def init_bracket(preset_name: str, custom_selections: list[tuple[str, str]] | None = None) -> None:
    st.session_state["bracket_round_of_16"] = []
    st.session_state["bracket_quarter_finals"] = []
    st.session_state["bracket_semi_finals"] = []
    st.session_state["bracket_final"] = []
    st.session_state["bracket_champion"] = None
    
    if preset_name == "2022 World Cup actuals":
        pairings = [
            ("Netherlands", "United States"),
            ("Argentina", "Australia"),
            ("Japan", "Croatia"),
            ("Brazil", "South Korea"),
            ("England", "Senegal"),
            ("France", "Poland"),
            ("Morocco", "Spain"),
            ("Portugal", "Switzerland")
        ]
    elif preset_name == "2026 Elo favorites":
        # Get WC 2026 teams sorted by Elo snapshot
        nations_names = list(name_to_iso.keys()) if name_to_iso else teams_list_all[:48]
        elo_list = []
        for name in nations_names:
            snap = predictor._team_snapshots.get(name)
            elo = float(snap["team_elo"]) if snap is not None and "team_elo" in snap else 1500.0
            elo_list.append((name, elo))
        elo_list.sort(key=lambda x: x[1], reverse=True)
        top_16 = [x[0] for x in elo_list[:16]]
        
        # Standard seeded pairing:
        # 1v16, 8v9, 4v13, 5v12, 2v15, 7v10, 3v14, 6v11
        pairings = [
            (top_16[0], top_16[15]),
            (top_16[7], top_16[8]),
            (top_16[3], top_16[12]),
            (top_16[4], top_16[11]),
            (top_16[1], top_16[14]),
            (top_16[6], top_16[9]),
            (top_16[2], top_16[13]),
            (top_16[5], top_16[10])
        ]
    else:  # Custom
        pairings = custom_selections or []

    for ta, tb in pairings:
        wdl = predictor.predict_fast(ta, tb, stage="round_of_16", is_home=0)
        st.session_state["bracket_round_of_16"].append({
            "team_a": ta,
            "team_b": tb,
            "winner": None,
            "score_a": None,
            "score_b": None,
            "pens_a": None,
            "pens_b": None,
            "prob_a": wdl["probabilities"]["win"],
            "prob_b": wdl["probabilities"]["loss"],
            "prob_draw": wdl["probabilities"]["draw"],
        })
    st.session_state["bracket_current_round"] = "r16"


def simulate_current_round() -> None:
    current = st.session_state["bracket_current_round"]
    if current == "r16":
        matches = st.session_state["bracket_round_of_16"]
        next_key = "qf"
    elif current == "qf":
        matches = st.session_state["bracket_quarter_finals"]
        next_key = "sf"
    elif current == "sf":
        matches = st.session_state["bracket_semi_finals"]
        next_key = "f"
    elif current == "f":
        matches = st.session_state["bracket_final"]
        next_key = "done"
    else:
        return

    from src.xg_engine import load_xg_model, simulate_match
    from src.statsbomb_shots import load_statsbomb_shots
    
    rng = np.random.default_rng()
    try:
        shots = load_statsbomb_shots()
        xg_model = load_xg_model()
    except Exception:
        shots = None
        xg_model = None

    winners = []
    for m in matches:
        if m["winner"] is not None:
            winners.append(m["winner"])
            continue

        ta = m["team_a"]
        tb = m["team_b"]

        # 1. Simulate regular-time score using xG or Elo proxy
        xg_sim = None
        if shots is not None and xg_model is not None and not shots.empty:
            try:
                xg_sim = simulate_match(ta, tb, n_simulations=500, shots=shots, model=xg_model)
            except Exception:
                pass

        if xg_sim and xg_sim.get("expected_xg_a") is not None:
            exp_a = xg_sim["expected_xg_a"]
            exp_b = xg_sim["expected_xg_b"]
        else:
            snap_a = predictor._team_snapshots.get(ta)
            snap_b = predictor._team_snapshots.get(tb)
            elo_a = float(snap_a["team_elo"]) if snap_a is not None and "team_elo" in snap_a else 1500.0
            elo_b = float(snap_b["team_elo"]) if snap_b is not None and "team_elo" in snap_b else 1500.0
            diff = elo_a - elo_b
            exp_a = max(0.4, 1.25 + diff / 420.0)
            exp_b = max(0.4, 1.25 - diff / 420.0)

        goals_a = int(rng.poisson(exp_a))
        goals_b = int(rng.poisson(exp_b))
        m["score_a"] = goals_a
        m["score_b"] = goals_b

        # 2. Enforce winner
        if goals_a > goals_b:
            m["winner"] = ta
        elif goals_b > goals_a:
            m["winner"] = tb
        else:
            # Draw -> Penalty shootout!
            # Get keeper save modifications
            snap_a = predictor._team_snapshots.get(ta)
            snap_b = predictor._team_snapshots.get(tb)
            
            p_winner, p_log = simulate_one_shootout(
                team_a_takers=[0.0]*5,
                team_b_takers=[0.0]*5,
                keeper_a_save_modifier=0.0,
                keeper_b_save_modifier=0.0,
                rng=rng,
                first_kicker="a"
            )
            m["pens_a"] = p_log[-1]["score_a"]
            m["pens_b"] = p_log[-1]["score_b"]
            m["winner"] = ta if p_winner == "a" else tb

        winners.append(m["winner"])

    # Build next round pairings
    if next_key == "qf":
        for i in range(0, len(winners), 2):
            w1, w2 = winners[i], winners[i+1]
            wdl = predictor.predict_fast(w1, w2, stage="quarter_final", is_home=0)
            st.session_state["bracket_quarter_finals"].append({
                "team_a": w1, "team_b": w2,
                "winner": None, "score_a": None, "score_b": None, "pens_a": None, "pens_b": None,
                "prob_a": wdl["probabilities"]["win"],
                "prob_b": wdl["probabilities"]["loss"],
                "prob_draw": wdl["probabilities"]["draw"],
            })
    elif next_key == "sf":
        for i in range(0, len(winners), 2):
            w1, w2 = winners[i], winners[i+1]
            wdl = predictor.predict_fast(w1, w2, stage="semi_final", is_home=0)
            st.session_state["bracket_semi_finals"].append({
                "team_a": w1, "team_b": w2,
                "winner": None, "score_a": None, "score_b": None, "pens_a": None, "pens_b": None,
                "prob_a": wdl["probabilities"]["win"],
                "prob_b": wdl["probabilities"]["loss"],
                "prob_draw": wdl["probabilities"]["draw"],
            })
    elif next_key == "f":
        w1, w2 = winners[0], winners[1]
        wdl = predictor.predict_fast(w1, w2, stage="final", is_home=0)
        st.session_state["bracket_final"].append({
            "team_a": w1, "team_b": w2,
            "winner": None, "score_a": None, "score_b": None, "pens_a": None, "pens_b": None,
            "prob_a": wdl["probabilities"]["win"],
            "prob_b": wdl["probabilities"]["loss"],
            "prob_draw": wdl["probabilities"]["draw"],
        })
    elif next_key == "done":
        st.session_state["bracket_champion"] = winners[0]

    st.session_state["bracket_current_round"] = next_key


# ---------------------------------------------------------------------------
# Setup sidebar and controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Bracket Settings")
    preset = st.selectbox(
        "Tournament Preset",
        ["2022 World Cup actuals", "2026 Elo favorites", "Custom Selection"]
    )
    
    custom_pairings = []
    if preset == "Custom Selection":
        st.subheader("Customize Round of 16")
        for i in range(1, 9):
            st.markdown(f"**Match {i}**")
            t_a = st.selectbox(f"Team {i}A", teams_list_all, index=min(i * 2, len(teams_list_all)-1), key=f"cust_ta_{i}")
            t_b = st.selectbox(f"Team {i}B", [t for t in teams_list_all if t != t_a], index=min(i * 2 + 1, len(teams_list_all)-1), key=f"cust_tb_{i}")
            custom_pairings.append((t_a, t_b))
            
    c_init, c_reset = st.columns(2)
    with c_init:
        init_btn = st.button("Initialize", type="primary", use_container_width=True)
    with c_reset:
        reset_btn = st.button("Reset state", use_container_width=True)

if reset_btn:
    st.session_state["bracket_current_round"] = "uninitialized"
    st.rerun()

if init_btn or st.session_state["bracket_current_round"] == "uninitialized":
    init_bracket(preset, custom_pairings if preset == "Custom Selection" else None)
    st.rerun()


# ---------------------------------------------------------------------------
# Visual display
# ---------------------------------------------------------------------------

curr_round = st.session_state["bracket_current_round"]

col_sim1, col_sim2 = st.columns([1, 1])
with col_sim1:
    sim_next_btn = st.button(
        "⚡ Simulate Next Round", 
        disabled=curr_round == "done", 
        type="primary", 
        use_container_width=True
    )
with col_sim2:
    sim_all_btn = st.button(
        "🏆 Simulate Entire Bracket", 
        disabled=curr_round == "done", 
        use_container_width=True
    )

if sim_next_btn:
    simulate_current_round()
    st.rerun()

if sim_all_btn:
    while st.session_state["bracket_current_round"] != "done":
        simulate_current_round()
    st.rerun()


# Render HTML/CSS styled bracket
def render_match_html(m: dict) -> str:
    ta = m["team_a"]
    tb = m["team_b"]
    wa = m["winner"]
    sa = m["score_a"]
    sb = m["score_b"]
    pa = m["pens_a"]
    pb = m["pens_b"]
    
    # Class styles based on results
    class_a = ""
    class_b = ""
    score_display_a = ""
    score_display_b = ""
    
    if wa is not None:
        if wa == ta:
            class_a = "winner-team"
            class_b = "loser-team"
        else:
            class_a = "loser-team"
            class_b = "winner-team"
            
        score_display_a = f"{sa}"
        score_display_b = f"{sb}"
        if pa is not None and pb is not None:
            score_display_a += f" <span class='pens-score'>({pa})</span>"
            score_display_b += f" <span class='pens-score'>({pb})</span>"
            
        score_html_a = f"<div class='score-badge'>{score_display_a}</div>"
        score_html_b = f"<div class='score-badge'>{score_display_b}</div>"
    else:
        score_html_a = f"<div class='prob-badge'>{m['prob_a']:.0%}</div>"
        score_html_b = f"<div class='prob-badge'>{m['prob_b']:.0%}</div>"

    html = f"""
    <div class="match-box">
        <div class="team-item {class_a}">
            <div class="team-name-wrap">
                {get_flag_html(ta)}
                <span>{ta}</span>
            </div>
            {score_html_a}
        </div>
        <div class="team-item {class_b}">
            <div class="team-name-wrap">
                {get_flag_html(tb)}
                <span>{tb}</span>
            </div>
            {score_html_b}
        </div>
    </div>
    """
    return html


# CSS styling
bracket_css = """
<style>
.bracket-wrapper {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 40px 20px;
    width: 100%;
    overflow-x: auto;
    min-height: 800px;
}
.round-col {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    height: 720px;
    min-width: 200px;
    margin: 0 10px;
}
.match-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin: 15px 0;
}
.team-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    font-size: 13px;
    color: #c9d1d9;
    border-bottom: 1px solid rgba(255,255,255,0.02);
}
.team-item:last-child {
    border-bottom: none;
}
.team-name-wrap {
    display: flex;
    align-items: center;
}
.flag-img {
    width: 22px;
    height: 15px;
    margin-right: 10px;
    border-radius: 2px;
    object-fit: cover;
    border: 1px solid rgba(255,255,255,0.15);
}
.score-badge {
    font-weight: 800;
    font-size: 12px;
    color: #ffffff;
    background: #21262d;
    padding: 3px 8px;
    border-radius: 4px;
    min-width: 22px;
    text-align: center;
}
.prob-badge {
    font-size: 10px;
    color: #58a6ff;
    background: rgba(56,139,253,0.15);
    padding: 2px 6px;
    border-radius: 4px;
}
.team-item.winner-team {
    color: #56d364;
    font-weight: 700;
}
.team-item.loser-team {
    color: #8b949e;
    text-decoration: line-through;
}
.pens-score {
    font-size: 9px;
    color: #8b949e;
    margin-left: 2px;
}
.champ-box {
    background: linear-gradient(135deg, #1f1b4d 0%, #0d0c24 100%);
    border: 2px solid #58a6ff;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(88,166,255,0.25);
    min-width: 180px;
}
.champ-title {
    font-size: 11px;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
    font-weight: 700;
}
.champ-name {
    font-size: 16px;
    color: #ffffff;
    font-weight: 800;
    margin-top: 6px;
}
</style>
"""

# Build bracket layout columns
r16_matches = st.session_state["bracket_round_of_16"]
qf_matches = st.session_state["bracket_quarter_finals"]
sf_matches = st.session_state["bracket_semi_finals"]
f_matches = st.session_state["bracket_final"]
champ = st.session_state["bracket_champion"]

# Round of 16 Column
html_r16 = '<div class="round-col">'
for m in r16_matches:
    html_r16 += render_match_html(m)
html_r16 += '</div>'

# Quarter-finals Column
html_qf = '<div class="round-col">'
if qf_matches:
    for m in qf_matches:
        html_qf += render_match_html(m)
else:
    for _ in range(4):
        html_qf += '<div class="match-box"><div class="team-item"><span>TBD</span></div><div class="team-item"><span>TBD</span></div></div>'
html_qf += '</div>'

# Semi-finals Column
html_sf = '<div class="round-col">'
if sf_matches:
    for m in sf_matches:
        html_sf += render_match_html(m)
else:
    for _ in range(2):
        html_sf += '<div class="match-box"><div class="team-item"><span>TBD</span></div><div class="team-item"><span>TBD</span></div></div>'
html_sf += '</div>'

# Final Column
html_f = '<div class="round-col">'
if f_matches:
    for m in f_matches:
        html_f += render_match_html(m)
else:
    html_f += '<div class="match-box"><div class="team-item"><span>TBD</span></div><div class="team-item"><span>TBD</span></div></div>'
html_f += '</div>'

# Champion Column
html_champ = '<div class="round-col" style="justify-content:center;">'
if champ:
    html_champ += f"""
    <div class="champ-box">
        <div class="champ-title">👑 Champion</div>
        {get_flag_html(champ).replace("width: 22px; height: 15px;", "width: 50px; height: 33px; margin: 10px 0;")}
        <div class="champ-name">{champ}</div>
    </div>
    """
else:
    html_champ += f"""
    <div class="champ-box" style="border-style: dashed; border-color: #30363d; background: transparent; box-shadow: none;">
        <div class="champ-title" style="color:#8b949e;">👑 Champion</div>
        <div style="font-size: 32px; margin: 10px 0; color:#30363d;">🏆</div>
        <div class="champ-name" style="color: #8b949e; font-size: 13px;">TBD</div>
    </div>
    """
html_champ += '</div>'

# Combine everything
full_bracket_html = f"""
{bracket_css}
<div class="bracket-wrapper">
    {html_r16}
    {html_qf}
    {html_sf}
    {html_f}
    {html_champ}
</div>
"""

st.markdown(full_bracket_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Deep dive panel for matches
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔍 Match Detail Deep Dive")
st.caption("Select any matchup from the simulated bracket below to see historical features or expected goals details.")

all_matches_flat = []
for m in r16_matches:
    all_matches_flat.append(m)
for m in qf_matches:
    all_matches_flat.append(m)
for m in sf_matches:
    all_matches_flat.append(m)
for m in f_matches:
    all_matches_flat.append(m)

if not all_matches_flat:
    st.info("Initialize the bracket first to inspect match details.")
else:
    match_options = [f"{m['team_a']} vs {m['team_b']}" for m in all_matches_flat]
    selected_opt = st.selectbox("Select match to analyze", match_options)
    selected_idx = match_options.index(selected_opt)
    m = all_matches_flat[selected_idx]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {m['team_a']} vs {m['team_b']}")
        if m['winner'] is not None:
            pens_txt = f" (after penalty shootout {m['pens_a']}-{m['pens_b']})" if m['pens_a'] is not None else ""
            st.success(f"**Winner: {m['winner']}**{pens_txt}")
            st.write(f"Simulated full-time score: **{m['score_a']} – {m['score_b']}**")
        else:
            st.info("Match has not been simulated yet.")
            
        st.markdown("**Model W/D/L probabilities:**")
        st.write(f"- {m['team_a']} Win: **{m['prob_a']:.1%}**")
        st.write(f"- Draw: **{m['prob_draw']:.1%}**")
        st.write(f"- {m['team_b']} Win: **{m['prob_b']:.1%}**")
        
    with col2:
        snap_a = predictor._team_snapshots.get(m['team_a'])
        snap_b = predictor._team_snapshots.get(m['team_b'])
        elo_a = float(snap_a["team_elo"]) if snap_a is not None and "team_elo" in snap_a else 1500.0
        elo_b = float(snap_b["team_elo"]) if snap_b is not None and "team_elo" in snap_b else 1500.0
        
        st.markdown("**Team Elo Ratings:**")
        st.metric(m['team_a'], f"{elo_a:.0f}")
        st.metric(m['team_b'], f"{elo_b:.0f}")
        
        diff = elo_a - elo_b
        st.caption(f"Elo difference: {diff:+.0f} points favoring {m['team_a'] if diff > 0 else m['team_b']}.")
