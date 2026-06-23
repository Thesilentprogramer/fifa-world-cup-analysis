"""Phase 3 — Player Dashboard.

Sections:
1. All-Time International Top Scorers (martj42 goalscorers.csv)
2. StatsBomb Player Event Aggregates (shots, goals, xG, xG chain, assists, key passes…)
3. Player Profile — stat cards + shot map
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.player_stats import aggregate_statsbomb_player_stats, load_top_scorers
from src.data_loader import INTERNATIONAL_DIR
from app.theme import inject_app_styles
from app.utils.player_photos import get_player_photo

inject_app_styles()

# ── extra CSS for the player photo card ──────────────────────────────────────
st.markdown("""
<style>
.photo-card {
    background: linear-gradient(145deg, rgba(108,99,255,0.12), rgba(30,40,70,0.6));
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 16px;
    padding: 1.2rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    backdrop-filter: blur(6px);
}
.photo-card img {
    border-radius: 12px;
    width: 100%;
    max-width: 240px;
    object-fit: cover;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    border: 2px solid rgba(108,99,255,0.4);
}
.photo-card .player-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    text-align: center;
    line-height: 1.2;
}
.photo-card .player-team {
    font-size: 0.88rem;
    color: rgba(255,215,0,0.85);
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.photo-source-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    text-align: center;
    margin-top: 0.15rem;
}
</style>
""", unsafe_allow_html=True)

st.title("⚽ Player Dashboard")
st.caption("All-time top scorers · StatsBomb event analytics · xG chain · player profiles")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_top_scorers(min_year, max_year, team, include_pens, top_n):
    return load_top_scorers(
        min_year=min_year or None,
        max_year=max_year or None,
        team=team or None,
        include_penalties=include_pens,
        top_n=top_n,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_statsbomb_stats():
    return aggregate_statsbomb_player_stats()


# ---------------------------------------------------------------------------
# Section 1: All-Time Top Scorers
# ---------------------------------------------------------------------------

st.header("🏆 All-Time International Top Scorers")
st.caption("Source: martj42/international_results (goalscorers.csv · updated daily)")

goalscorers_path = INTERNATIONAL_DIR / "goalscorers.csv"
if not goalscorers_path.exists():
    st.warning(
        "Goalscorers data not found. Run:\n"
        "```bash\npython scripts/download_data.py --international-only\n```"
    )
else:
    # Sidebar filters
    with st.sidebar:
        st.header("🎯 Top Scorers Filters")

        gs_raw = pd.read_csv(goalscorers_path, parse_dates=["date"], low_memory=False)
        gs_raw["year"] = gs_raw["date"].dt.year
        all_teams_gs = sorted(gs_raw["team"].dropna().unique().tolist())
        min_yr = int(gs_raw["year"].min())
        max_yr = int(gs_raw["year"].max())

        year_range = st.slider("Year range", min_yr, max_yr, (2000, max_yr))
        sel_team = st.selectbox("Filter by team (optional)", ["All teams"] + all_teams_gs)
        include_pens = st.toggle("Include penalty goals", value=True)
        top_n = st.slider("Show top N scorers", 10, 100, 25)

    team_filter = None if sel_team == "All teams" else sel_team

    with st.spinner("Loading top scorers…"):
        scorers_df = _cached_top_scorers(
            year_range[0], year_range[1], team_filter, include_pens, top_n
        )

    if scorers_df.empty:
        st.info("No data for selected filters.")
    else:
        scorers_df.index = range(1, len(scorers_df) + 1)

        # Bar chart — top 20
        chart_df = scorers_df.head(20).copy()
        fig_bar = px.bar(
            chart_df,
            x="goals",
            y="scorer",
            orientation="h",
            color="goals",
            color_continuous_scale="viridis",
            hover_data=["team", "penalty_goals", "matches"],
            text="goals",
            labels={"goals": "Goals", "scorer": "Player"},
            title=f"Top {min(20, len(chart_df))} International Scorers",
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            height=max(300, len(chart_df) * 28),
            margin=dict(l=10, r=10, t=50, b=20),
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, width="stretch")

        # Table
        display_df = scorers_df.copy()
        display_df.columns = ["Player", "Team", "Goals", "Penalty Goals", "Matches"]
        display_df["Goals / Match"] = (display_df["Goals"] / display_df["Matches"]).round(2)
        st.dataframe(display_df, use_container_width=True, height=420)

        # Quick stat summary
        total_goals = scorers_df["goals"].sum()
        top_scorer = scorers_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Top scorer", top_scorer["scorer"])
        c2.metric("Goals", top_scorer["goals"])
        c3.metric(f"Goals in shown period ({year_range[0]}–{year_range[1]})", int(total_goals))


# ---------------------------------------------------------------------------
# Section 2: StatsBomb Player Event Aggregates
# ---------------------------------------------------------------------------

st.divider()
st.header("📊 StatsBomb Player Event Analytics")
st.caption(
    "Compiled from cached StatsBomb open-data events (World Cup, Euro, Copa América, AFCON). "
    "Covers ~369 international matches."
)

with st.spinner("Aggregating StatsBomb player events… (may take a moment first run)"):
    sb_df = _cached_statsbomb_stats()

if sb_df.empty:
    st.info(
        "No StatsBomb event data found. Run:\n"
        "```bash\npython scripts/download_data.py --statsbomb-only\n```\n"
        "This downloads match-level events (not the full event download which can be large)."
    )
else:
    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        all_comps = sorted(sb_df["competition"].unique().tolist())
        sel_comp = st.multiselect("Competition", all_comps, default=all_comps)
    with col_f2:
        all_sb_teams = sorted(sb_df["team"].unique().tolist())
        sel_sb_team = st.selectbox("Team (all)", ["All"] + all_sb_teams, key="sb_team")
    with col_f3:
        metric_choice = st.selectbox(
            "Rank by",
            ["goals", "xG", "xg_chain", "shots", "key_passes", "assists", "completed_passes",
             "dribbles", "tackles", "interceptions"],
        )

    filtered = sb_df.copy()
    if sel_comp:
        filtered = filtered[filtered["competition"].isin(sel_comp)]
    if sel_sb_team != "All":
        filtered = filtered[filtered["team"] == sel_sb_team]

    # Aggregate across competitions/seasons for the same player+team
    agg_cols = ["goals", "shots", "xG", "key_passes", "assists", "passes",
                "completed_passes", "dribbles", "completed_dribbles",
                "tackles", "interceptions", "xg_chain"]
    grouped = (
        filtered.groupby(["player", "team"])[agg_cols]
        .sum()
        .reset_index()
        .sort_values(metric_choice, ascending=False)
    )
    grouped["xG/Shot"] = (grouped["xG"] / grouped["shots"].clip(lower=1)).round(3)
    grouped["Pass%"] = (grouped["completed_passes"] / grouped["passes"].clip(lower=1) * 100).round(1)
    grouped["Dribble%"] = (grouped["completed_dribbles"] / grouped["dribbles"].clip(lower=1) * 100).round(1)

    top_players = grouped.head(25).reset_index(drop=True)
    top_players.index = range(1, len(top_players) + 1)

    # Chart
    chart_players = top_players.head(15)
    fig_sb = px.bar(
        chart_players,
        x=metric_choice,
        y="player",
        orientation="h",
        color="team",
        hover_data=["goals", "xG", "xg_chain", "assists", "key_passes"],
        text=metric_choice,
        title=f"Top 15 Players by {metric_choice}",
    )
    fig_sb.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(300, len(chart_players) * 30),
        margin=dict(l=10, r=10, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_sb.update_traces(textposition="outside")
    st.plotly_chart(fig_sb, width="stretch")

    # Full leaderboard table
    display_sb = top_players[["player", "team", "goals", "shots", "xG", "xG/Shot",
                               "xg_chain", "key_passes", "assists",
                               "passes", "Pass%", "dribbles", "Dribble%",
                               "tackles", "interceptions"]].copy()
    display_sb.columns = [
        "Player", "Team", "Goals", "Shots", "xG", "xG/Shot",
        "xG Chain", "Key Passes", "Assists",
        "Passes", "Pass%", "Dribbles", "Dribble%",
        "Tackles", "Interceptions",
    ]
    st.dataframe(display_sb, use_container_width=True, height=480)


# ---------------------------------------------------------------------------
# Section 3: Player Profile Detail
# ---------------------------------------------------------------------------

st.divider()
st.header("👤 Player Profile")

if sb_df.empty:
    st.info("StatsBomb data needed for player profiles.")
else:
    all_players = sorted(sb_df["player"].unique().tolist())
    sel_player = st.selectbox("Select a player", all_players, key="profile_player")

    player_rows = sb_df[sb_df["player"] == sel_player]
    if player_rows.empty:
        st.info("No data for selected player.")
    else:
        # Aggregate all competitions
        num_cols = ["goals", "shots", "xG", "key_passes", "assists", "passes",
                    "completed_passes", "dribbles", "completed_dribbles",
                    "tackles", "interceptions", "xg_chain"]
        agg = player_rows[num_cols].sum()
        team = player_rows["team"].iloc[0]

        # ── Fetch player photo (API-Football → Wikipedia fallback) ──────────
        with st.spinner("Loading player photo…"):
            photo_url = get_player_photo(sel_player)

        # ── Layout: photo card (left) | stats (right) ───────────────────────
        col_photo, col_stats = st.columns([1, 2.8], gap="large")

        with col_photo:
            if photo_url:
                is_wiki = "wikipedia" in photo_url or "wikimedia" in photo_url
                source_label = "📷 Wikipedia (CC-BY-SA)" if is_wiki else "📷 API-Football"
                # st.image() is the correct way to render external URLs in Streamlit
                st.image(photo_url, use_container_width=True)
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-top:0.5rem;">
                        <div class="player-name">{sel_player}</div>
                        <div class="player-team">{team}</div>
                        <div class="photo-source-label">{source_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Placeholder when no photo is found
                st.markdown(
                    f"""
                    <div class="photo-card">
                        <div style="font-size:5rem;">⚽</div>
                        <div class="player-name">{sel_player}</div>
                        <div class="player-team">{team}</div>
                        <div class="photo-source-label">No photo available</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_stats:
            st.markdown(f"#### {sel_player} &nbsp;·&nbsp; {team}", unsafe_allow_html=True)

            # Stat cards — row 1
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("Goals", int(agg["goals"]))
            r1c2.metric("Shots", int(agg["shots"]))
            r1c3.metric("xG", f"{agg['xG']:.2f}")
            r1c4.metric("xG Chain", f"{agg['xg_chain']:.2f}")

            # Stat cards — row 2
            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            r2c1.metric("Key Passes", int(agg["key_passes"]))
            r2c2.metric("Assists", int(agg["assists"]))
            r2c3.metric("Tackles", int(agg["tackles"]))
            r2c4.metric("Interceptions", int(agg["interceptions"]))

            # Stats by competition
            if len(player_rows) > 1:
                st.subheader("By Competition")
                comp_agg = player_rows.groupby(["competition", "season"])[num_cols].sum().reset_index()
                comp_agg = comp_agg.sort_values("goals", ascending=False)
                comp_agg.columns = ["Competition", "Season"] + [c.title().replace("_", " ") for c in num_cols]
                st.dataframe(comp_agg.set_index("Competition"), use_container_width=True)

        # ── Radar chart (full width below photo + stats) ────────────────────
        radar_cols = ["goals", "shots", "xG", "key_passes", "assists", "tackles"]
        radar_vals = [float(agg[c]) for c in radar_cols]
        radar_labels = ["Goals", "Shots", "xG", "Key Passes", "Assists", "Tackles"]
        max_vals = [max(sb_df[c].max(), 1) for c in radar_cols]
        radar_norm = [v / m for v, m in zip(radar_vals, max_vals)]
        radar_norm.append(radar_norm[0])
        radar_labels_loop = radar_labels + [radar_labels[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_norm,
            theta=radar_labels_loop,
            fill="toself",
            line=dict(color="#6C63FF", width=2),
            fillcolor="rgba(108, 99, 255, 0.25)",
            name=sel_player,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"{sel_player} — Normalised Performance Radar",
            height=400,
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig_radar, width="stretch")

st.caption("StatsBomb open data covers select international tournaments only. Figures are event-level aggregates.")
