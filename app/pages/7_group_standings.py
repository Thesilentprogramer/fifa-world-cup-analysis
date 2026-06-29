"""Phase 5 — Group Standings & Monte Carlo Projections.

Calculates actual group tables dynamically from the fixture results
and uses Monte Carlo simulation to project final outcomes and qualification probabilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api_football_client import load_wc_fixtures
from src.match_predictor import get_predictor
from src.group_simulation import simulate_group_stage
from app.theme import inject_app_styles

# Page Config
st.set_page_config(
    page_title="FIFA World Cup Group Standings",
    page_icon="📊",
    layout="wide",
)

inject_app_styles()

st.title("📊 Group Standings & Projections")
st.caption(
    "Live dynamic standings · remaining fixtures Monte Carlo simulation · "
    "qualification probabilities · goal-difference tiebreaker resolution"
)

# Load predictor
try:
    predictor = get_predictor()
except FileNotFoundError as e:
    st.error(f"Model not found. Run `python scripts/train_model.py` first.\n\n{e}")
    st.stop()


def extract_groups_from_fixtures(fixtures_df: pd.DataFrame) -> dict[str, list[str]]:
    """Dynamically cluster teams into groups based on who plays each other in group stage."""
    group_matches = fixtures_df[fixtures_df["stage"] == "group"].copy()
    if group_matches.empty:
        return {}

    # Build adjacency list
    adj = {}
    for _, row in group_matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        adj.setdefault(h, set()).add(a)
        adj.setdefault(a, set()).add(h)

    visited = set()
    components = []

    for team in adj:
        if team not in visited:
            comp = []
            queue = [team]
            visited.add(team)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for opp in adj[curr]:
                    if opp not in visited:
                        visited.add(opp)
                        queue.append(opp)
            components.append(comp)

    # Sort components by alphabetical first team
    components = sorted(components, key=lambda c: sorted(c)[0])
    
    groups = {}
    for i, comp in enumerate(components):
        group_letter = chr(65 + i) if i < 26 else f"Group {i+1}"
        groups[f"Group {group_letter}"] = sorted(comp)
    return groups


def compute_actual_standings(teams: list[str], fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate actual standings for group from played fixtures."""
    finished_matches = fixtures_df[
        (fixtures_df["stage"] == "group") & 
        (fixtures_df["is_finished"] == True) & 
        (fixtures_df["home_team"].isin(teams)) & 
        (fixtures_df["away_team"].isin(teams))
    ]
    
    records = []
    for team in teams:
        played = won = drawn = lost = gf = ga = 0
        for _, row in finished_matches.iterrows():
            hs = int(row["home_score"]) if pd.notna(row["home_score"]) else 0
            aws = int(row["away_score"]) if pd.notna(row["away_score"]) else 0
            
            if row["home_team"] == team:
                played += 1
                gf += hs
                ga += aws
                if hs > aws:
                    won += 1
                elif hs < aws:
                    lost += 1
                else:
                    drawn += 1
            elif row["away_team"] == team:
                played += 1
                gf += aws
                ga += hs
                if aws > hs:
                    won += 1
                elif aws < hs:
                    lost += 1
                else:
                    drawn += 1
        
        records.append({
            "Team": team,
            "P": played,
            "W": won,
            "D": drawn,
            "L": lost,
            "GF": gf,
            "GA": ga,
            "GD": gf - ga,
            "Pts": won * 3 + drawn
        })
    df = pd.DataFrame(records)
    df = df.sort_values(by=["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True]).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


# 1. Load fixtures
fixtures = load_wc_fixtures()
if fixtures.empty:
    st.warning("No World Cup fixtures available in database.")
    st.stop()

# 2. Extract groups
groups = extract_groups_from_fixtures(fixtures)
if not groups:
    st.warning("No group stage matches found in fixtures.")
    st.stop()

# 3. Sidebar selection
selected_group = st.selectbox("Select Group", list(groups.keys()))
group_teams = groups[selected_group]

st.write(f"### ⚔️ {selected_group} Standings & Projections")

# Filter matches for the selected group
group_matches = fixtures[
    (fixtures["stage"] == "group") & 
    (fixtures["home_team"].isin(group_teams)) & 
    (fixtures["away_team"].isin(group_teams))
].copy()

finished_matches = group_matches[group_matches["is_finished"] == True].copy()
upcoming_matches = group_matches[group_matches["is_upcoming"] == True].copy()

col_tables, col_charts = st.columns([2, 1.2])

with col_tables:
    # 4. Actual table
    st.write("#### 📋 Actual Group Standings")
    actual_table = compute_actual_standings(group_teams, fixtures)
    st.dataframe(
        actual_table.set_index("Rank"),
        use_container_width=True,
    )

    # 5. Monte Carlo simulation
    st.write("#### 🔮 Projected Standings (5,000 runs Monte Carlo)")
    with st.spinner("Simulating remaining group matches..."):
        sim_results = simulate_group_stage(
            teams=group_teams,
            finished_matches=finished_matches,
            upcoming_matches=upcoming_matches,
            predictor=predictor,
            n_simulations=5000,
        )

    # Convert results dict to dataframe
    sim_rows = []
    for team in group_teams:
        r = sim_results[team]
        sim_rows.append({
            "Team": team,
            "Projected Pts": f"{r['avg_pts']:.1f}",
            "Projected GD": f"{r['avg_gd']:.1f}",
            "1st (%)": f"{r['prob_1st']:.1%}",
            "2nd (%)": f"{r['prob_2nd']:.1%}",
            "3rd (%)": f"{r['prob_3rd']:.1%}",
            "4th (%)": f"{r['prob_4th']:.1%}",
            "Qualify (%)": f"{r['prob_qualify']:.1%}"
        })
    
    sim_df = pd.DataFrame(sim_rows)
    # Sort sim_df by qualify percent / projected points
    sim_df["sort_val"] = sim_df["Projected Pts"].astype(float)
    sim_df = sim_df.sort_values("sort_val", ascending=False).drop(columns=["sort_val"]).reset_index(drop=True)
    sim_df.insert(0, "Proj. Rank", range(1, len(sim_df) + 1))
    
    st.dataframe(
        sim_df.set_index("Proj. Rank"),
        use_container_width=True,
    )

with col_charts:
    st.write("#### 📈 Qualification Probabilities")
    
    qual_probs = [sim_results[t]["prob_qualify"] for t in sim_df["Team"]]
    teams_sorted = list(sim_df["Team"])
    
    fig = go.Figure(go.Bar(
        x=qual_probs,
        y=teams_sorted,
        orientation="h",
        marker=dict(
            color=qual_probs,
            colorscale="Viridis",
            line=dict(color="rgba(255, 255, 255, 0.5)", width=1)
        ),
        text=[f"{p:.1%}" for p in qual_probs],
        textposition="outside"
    ))
    fig.update_layout(
        xaxis=dict(tickformat=".0%", range=[0, 1.1]),
        yaxis=dict(autorange="reversed"),
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Progress/status of the group
    n_finished = len(finished_matches)
    n_upcoming = len(upcoming_matches)
    n_total = n_finished + n_upcoming
    
    st.metric("Matches Played", f"{n_finished} / {n_total}", help="Total matches played in this group stage")
    if n_upcoming == 0:
        st.success("🎉 Group stage completed! Final standings are locked.")
    else:
        st.info(f"🔮 {n_upcoming} matches remaining. Projected standings are simulated using match predictions.")

# 6. Group Fixtures section
st.divider()
st.write("### 📅 Group Fixtures & Results")

fix_finished, fix_upcoming = st.columns(2)

with fix_finished:
    st.write("##### Played Matches")
    if finished_matches.empty:
        st.write("*No matches played yet.*")
    else:
        for _, row in finished_matches.sort_values("match_date").iterrows():
            st.markdown(
                f"⚽ **{row['home_team']}** `{int(row['home_score'])} - {int(row['away_score'])}` **{row['away_team']}**  \n"
                f"<small>Date: {pd.to_datetime(row['match_date']).strftime('%b %d, %Y')} · Venue: {row['venue']}</small>",
                unsafe_allow_html=True
            )

with fix_upcoming:
    st.write("##### Remaining Matches")
    if upcoming_matches.empty:
        st.write("*No matches remaining.*")
    else:
        for _, row in upcoming_matches.sort_values("match_date").iterrows():
            # Get prediction probabilities
            pred = predictor.predict_fast(row['home_team'], row['away_team'], stage="group", is_home=0)
            probs = pred["probabilities"]
            st.markdown(
                f"🔮 **{row['home_team']}** vs **{row['away_team']}**  \n"
                f"<small>P({row['home_team']}/D/{row['away_team']}): {probs['win']:.0%} / {probs['draw']:.0%} / {probs['loss']:.0%}</small>  \n"
                f"<small>Date: {pd.to_datetime(row['match_date']).strftime('%b %d, %H:%M UTC')} · Venue: {row['venue']}</small>",
                unsafe_allow_html=True
            )
