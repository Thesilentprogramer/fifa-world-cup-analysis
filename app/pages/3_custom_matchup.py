"""Custom head-to-head matchup predictor.

Phase 5 addition: shows Polymarket / bookmaker implied probabilities
side-by-side with model output.
Enhanced additions:
- Detailed side-by-side team metrics comparison.
- Top scorers comparison from historical data.
- Built-in penalty shootout simulator tiebreaker analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api_football_client import load_wc_fixtures, fetch_live_odds_for_fixture, parse_match_winner_odds
from src.odds_comparison import odds_to_implied_probs
from src.pdf_generator import generate_prematch_pdf
from src.match_predictor import get_predictor
from src.prematch_analysis import analyze_fixture
from src.player_stats import load_top_scorers
from src.penalty_engine import KEEPER_PRESETS, TAKER_PRESETS, simulate_shootout
from app.theme import inject_app_styles

inject_app_styles()

st.title("⚽ H2H Matchup Center")
st.caption("Pick any two national teams · prediction & SHAP · metrics analysis · top scorers · penalty shootouts")


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
    is_home = st.radio("Team A plays at", ["Home", "Neutral/Away"], index=1)
    predict_btn = st.button("Predict Matchup", type="primary", use_container_width=True)


def find_keeper_preset(team_name: str) -> str:
    for name in KEEPER_PRESETS.keys():
        if team_name.lower() in name.lower():
            return name
    return "Average keeper"


if predict_btn or "h2h_last_predicted" in st.session_state and st.session_state.h2h_last_predicted == (team_a, team_b, stage, is_home):
    # Keep track of prediction in session state to prevent losing it on tab switches
    st.session_state.h2h_last_predicted = (team_a, team_b, stage, is_home)
    
    result = predictor.predict(
        team_a, team_b, stage=stage, is_home=1 if is_home == "Home" else 0,
    )
    analysis = analyze_fixture(team_a, team_b, stage=stage, predictor=predictor, n_xg_sims=3000)
    probs = result["probabilities"]

    # Use tabs for structured layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Prediction & SHAP",
        "📊 H2H Team Metrics",
        "⚽ Top Scorers",
        "🥅 Penalty Tiebreaker"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: PREDICTION & SHAP
    # -------------------------------------------------------------------------
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Match Outcome Probability")
            outcome = result["predicted_outcome"]
            if outcome == "Win":
                st.success(f"Predicted: **{team_a} Win** ({result['confidence']:.0%})")
            elif outcome == "Loss":
                st.success(f"Predicted: **{team_b} Win** ({result['confidence']:.0%})")
            else:
                st.warning(f"Predicted: **Draw** ({result['confidence']:.0%})")

            labels = [f"{team_a} Win", "Draw", f"{team_b} Win"]
            values = [probs["win"], probs["draw"], probs["loss"]]
            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=0.45,
                marker=dict(colors=["#2E7D32", "#F9A825", "#C62828"]),
                textinfo="label+percent",
            )])
            fig.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, width="stretch")

        with col2:
            if analysis["expected_xg_home"] is not None:
                st.subheader("xG Simulation Scoreline")
                st.metric(team_a, f"{analysis['expected_xg_home']:.2f} xG")
                st.metric(team_b, f"{analysis['expected_xg_away']:.2f} xG")
                if analysis["top_scoreline"]:
                    st.caption(f"Likeliest score: **{analysis['top_scoreline']}** ({analysis['top_scoreline_prob']:.0%} chance)")

            st.subheader("Why this prediction?")
            st.markdown(result.get("narrative", ""))
            shap_data = result.get("shap", [])
            if shap_data:
                features = [d["label"] for d in shap_data][::-1]
                values = [d["shap_value"] for d in shap_data][::-1]
                colors_bar = ["#2E7D32" if v > 0 else "#C62828" for v in values]
                fig_shap = go.Figure(go.Bar(x=values, y=features, orientation="h", marker_color=colors_bar))
                fig_shap.update_layout(title="Top 5 SHAP Features", height=280, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_shap, width="stretch")

        # Market Odds Comparison
        st.divider()
        st.subheader("📈 Model vs Market Odds")

        snap_a = predictor._team_snapshots.get(team_a)
        snap_b = predictor._team_snapshots.get(team_b)

        poly_win = snap_a.get("polymarket_prob_win") if snap_a is not None else None
        poly_draw = snap_a.get("polymarket_prob_draw") if snap_a is not None else None
        poly_loss = snap_a.get("polymarket_prob_loss") if snap_a is not None else None

        model_row = {
            "Source": "🤖 Our Model",
            f"{team_a} Win": f"{probs['win']:.1%}",
            "Draw": f"{probs['draw']:.1%}",
            f"{team_b} Win": f"{probs['loss']:.1%}",
        }
        rows = [model_row]

        if poly_win and pd.notna(poly_win) and float(poly_win) > 0:
            rows.append({
                "Source": "📊 Polymarket (implied)",
                f"{team_a} Win": f"{float(poly_win):.1%}",
                "Draw": f"{float(poly_draw):.1%}" if poly_draw and pd.notna(poly_draw) else "N/A",
                f"{team_b} Win": f"{float(poly_loss):.1%}" if poly_loss and pd.notna(poly_loss) else "N/A",
            })

        # Fetch live odds from API-Football if available for this head-to-head match
        live_odds_added = False
        try:
            fixtures = load_wc_fixtures()
            if not fixtures.empty:
                matches = fixtures[
                    ((fixtures["home_team"] == team_a) & (fixtures["away_team"] == team_b)) |
                    ((fixtures["home_team"] == team_b) & (fixtures["away_team"] == team_a))
                ]
                if not matches.empty:
                    # Sort upcoming first, or just take first
                    match_row = matches.iloc[0]
                    fixture_id = match_row.get("fixture_id")
                    if fixture_id:
                        raw_odds = fetch_live_odds_for_fixture(int(fixture_id))
                        if raw_odds:
                            parsed_odds = parse_match_winner_odds(raw_odds, team_a, team_b)
                            if parsed_odds:
                                dec_home = parsed_odds["home_odds"]
                                dec_draw = parsed_odds["draw_odds"]
                                dec_away = parsed_odds["away_odds"]
                                # If team_a was away in the actual fixture, flip odds
                                is_a_home = (match_row["home_team"] == team_a)
                                implied = odds_to_implied_probs(dec_home, dec_draw, dec_away)
                                if not is_a_home:
                                    implied = implied[::-1] # reverse so a_win, draw, b_win
                                rows.append({
                                    "Source": f"🏦 {parsed_odds['bookmaker']} (Live)",
                                    f"{team_a} Win": f"{implied[0]:.1%}",
                                    "Draw": f"{implied[1]:.1%}",
                                    f"{team_b} Win": f"{implied[2]:.1%}",
                                })
                                live_odds_added = True
        except Exception:
            pass

        bk_win = snap_a.get("implied_prob_win") if snap_a is not None else None
        bk_draw = snap_a.get("implied_prob_draw") if snap_a is not None else None
        bk_loss = snap_a.get("implied_prob_loss") if snap_a is not None else None
        if not live_odds_added and bk_win and pd.notna(bk_win) and float(bk_win) > 0:
            rows.append({
                "Source": "🏦 Bookmaker (Hist)",
                f"{team_a} Win": f"{float(bk_win):.1%}",
                "Draw": f"{float(bk_draw):.1%}" if bk_draw and pd.notna(bk_draw) else "N/A",
                f"{team_b} Win": f"{float(bk_loss):.1%}" if bk_loss and pd.notna(bk_loss) else "N/A",
            })

        comparison_df = pd.DataFrame(rows)
        st.table(comparison_df.set_index("Source"))

        if len(rows) >= 2:
            sources = [r["Source"] for r in rows]
            win_vals = []
            for r in rows:
                v = r[f"{team_a} Win"].replace("%", "")
                try:
                    win_vals.append(float(v) / 100)
                except Exception:
                    win_vals.append(0.0)

            fig_comp = go.Figure()
            colors = ["#6C63FF", "#F59E0B", "#10B981"]
            for i, (src, wv) in enumerate(zip(sources, win_vals)):
                fig_comp.add_trace(go.Bar(
                    name=src, x=[src], y=[wv],
                    marker_color=colors[i % len(colors)],
                    text=[f"{wv:.1%}"], textposition="outside",
                ))
            fig_comp.update_layout(
                title=f"{team_a} Win Probability Comparison",
                yaxis=dict(tickformat=".0%", range=[0, 1.1]),
                height=260,
                showlegend=False,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_comp, width="stretch")

            # One-click PDF briefing export
            try:
                pdf_data = generate_prematch_pdf(
                    team_a, team_b,
                    {
                        "stage": stage,
                        "venue": "Neutral Venue",
                        "prob_home_win": analysis["prob_home_win"],
                        "prob_draw": analysis["prob_draw"],
                        "prob_away_win": analysis["prob_away_win"],
                        "expected_xg_home": analysis.get("expected_xg_home"),
                        "expected_xg_away": analysis.get("expected_xg_away"),
                        "top_scoreline": analysis.get("top_scoreline"),
                        "elo_home": analysis.get("elo_home"),
                        "elo_away": analysis.get("elo_away"),
                        "narrative": analysis.get("narrative", "")
                    }
                )
                st.download_button(
                    label="📄 Download Pre-Match Briefing PDF",
                    data=pdf_data,
                    file_name=f"{team_a}_vs_{team_b}_briefing.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"pdf_btn_custom_{team_a}_{team_b}"
                )
            except Exception as e:
                st.caption(f"⚠️ PDF generation failed: {e}")


    # -------------------------------------------------------------------------
    # TAB 2: H2H TEAM METRICS
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("📊 Side-by-Side Team Performance Metrics")
        st.caption("Latest ratings and historical aggregates derived from raw match results.")
        
        snap_a = predictor._team_snapshots.get(team_a)
        snap_b = predictor._team_snapshots.get(team_b)

        metrics_compare = {
            "Performance Indicator": [
                "Elo Rating",
                "Squad Market Value (Transfermarkt)",
                "Goals Scored per Game (last 5)",
                "Goals Conceded per Game (last 5)",
                "Average Possession % (last 5)",
                "Average expected Goals (xG) (last 5)",
                "Form points (last 5 matches)",
                "Rest Days (prior to match)"
            ],
            team_a: [
                f"{snap_a.get('team_elo', 1500.0):.0f}" if snap_a is not None else "1500",
                f"€{snap_a.get('squad_market_value', 0)/1e6:.1f}M" if snap_a is not None and snap_a.get('squad_market_value', 0) > 0 else "N/A",
                f"{snap_a.get('goals_for_avg_5', 0.0):.2f}" if snap_a is not None else "0.00",
                f"{snap_a.get('goals_against_avg_5', 0.0):.2f}" if snap_a is not None else "0.00",
                f"{snap_a.get('possession_pct_avg_5', 50.0):.1f}%" if snap_a is not None and snap_a.get('possession_pct_avg_5', 0) > 0 else "50.0%",
                f"{snap_a.get('xg_avg_5', 0.0):.2f}" if snap_a is not None and snap_a.get('xg_avg_5', 0) > 0 else "0.00",
                f"{snap_a.get('form_5', 0.0):.0f} / 15" if snap_a is not None else "0 / 15",
                f"{snap_a.get('team_rest_days', 7):.0f} days" if snap_a is not None else "7 days"
            ],
            team_b: [
                f"{snap_b.get('team_elo', 1500.0):.0f}" if snap_b is not None else "1500",
                f"€{snap_b.get('squad_market_value', 0)/1e6:.1f}M" if snap_b is not None and snap_b.get('squad_market_value', 0) > 0 else "N/A",
                f"{snap_b.get('goals_for_avg_5', 0.0):.2f}" if snap_b is not None else "0.00",
                f"{snap_b.get('goals_against_avg_5', 0.0):.2f}" if snap_b is not None else "0.00",
                f"{snap_b.get('possession_pct_avg_5', 50.0):.1f}%" if snap_b is not None and snap_b.get('possession_pct_avg_5', 0) > 0 else "50.0%",
                f"{snap_b.get('xg_avg_5', 0.0):.2f}" if snap_b is not None and snap_b.get('xg_avg_5', 0) > 0 else "0.00",
                f"{snap_b.get('form_5', 0.0):.0f} / 15" if snap_b is not None else "0 / 15",
                f"{snap_b.get('team_rest_days', 7):.0f} days" if snap_b is not None else "7 days"
            ]
        }
        df_metrics = pd.DataFrame(metrics_compare)
        st.dataframe(df_metrics.set_index("Performance Indicator"), use_container_width=True)

        # Elo and Squad Value Comparison Bar Chart
        if snap_a is not None and snap_b is not None:
            elo_a = float(snap_a.get('team_elo', 1500.0))
            elo_b = float(snap_b.get('team_elo', 1500.0))
            
            fig_elo = go.Figure(data=[
                go.Bar(name=team_a, x=["Elo rating"], y=[elo_a], marker_color="#6C63FF"),
                go.Bar(name=team_b, x=["Elo rating"], y=[elo_b], marker_color="#F59E0B")
            ])
            fig_elo.update_layout(title="Elo Rating Comparison", barmode='group', height=300)
            st.plotly_chart(fig_elo, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 3: TOP SCORERS
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("⚽ Historical International Top Scorers")
        st.caption("Top goalscorers for both nations loaded from `goalscorers.csv`.")
        
        col_sa, col_sb = st.columns(2)
        with col_sa:
            st.markdown(f"**{team_a} scorers**")
            sa_scorers = load_top_scorers(team=team_a, top_n=5)
            if sa_scorers.empty:
                st.info(f"No goalscorer records found for {team_a}.")
            else:
                sa_display = sa_scorers[["scorer", "goals", "penalty_goals", "matches"]].copy()
                sa_display.columns = ["Player", "Goals", "Penalties", "Matches"]
                st.dataframe(sa_display, use_container_width=True, hide_index=True)
                
        with col_sb:
            st.markdown(f"**{team_b} scorers**")
            sb_scorers = load_top_scorers(team=team_b, top_n=5)
            if sb_scorers.empty:
                st.info(f"No goalscorer records found for {team_b}.")
            else:
                sb_display = sb_scorers[["scorer", "goals", "penalty_goals", "matches"]].copy()
                sb_display.columns = ["Player", "Goals", "Penalties", "Matches"]
                st.dataframe(sb_display, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 4: PENALTY TIEBREAKER
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("🥅 Penalty Shootout Simulator")
        st.caption(
            "Predicts the outcome of the match if it goes to a penalty shootout tiebreaker. "
            "Runs 10,000 Monte Carlo simulations utilizing historical rates and goalkeeper penalty-saving presets."
        )

        keeper_a = find_keeper_preset(team_a)
        keeper_b = find_keeper_preset(team_b)

        st.markdown(
            f"**Simulating Shootout with default Keepers:**\n"
            f"- {team_a} Goalkeeper: **{keeper_a}** (Save modifier: {KEEPER_PRESETS[keeper_a]['save_modifier']:+})\n"
            f"- {team_b} Goalkeeper: **{keeper_b}** (Save modifier: {KEEPER_PRESETS[keeper_b]['save_modifier']:+})"
        )

        # Run 10k simulations
        with st.spinner("Simulating shootouts..."):
            ps_result = simulate_shootout(
                team_a_takers=[0.0]*5,  # Standard taker modifiers
                team_b_takers=[0.0]*5,
                keeper_a_save_modifier=KEEPER_PRESETS[keeper_a]["save_modifier"],
                keeper_b_save_modifier=KEEPER_PRESETS[keeper_b]["save_modifier"],
                n_simulations=10000,
                seed=42
            )

        win_a = ps_result["win_prob_a"]
        win_b = ps_result["win_prob_b"]

        col_pa, col_pb = st.columns(2)
        with col_pa:
            st.metric(f"{team_a} Shootout Win Chance", f"{win_a:.1%}")
        with col_pb:
            st.metric(f"{team_b} Shootout Win Chance", f"{win_b:.1%}")

        # Render Shootout Gauge Chart
        fig_ps = go.Figure()
        fig_ps.add_trace(go.Bar(
            name="Shootout Win %",
            x=[team_a, team_b],
            y=[win_a, win_b],
            marker_color=["#6C63FF", "#F59E0B"],
            text=[f"{win_a:.1%}", f"{win_b:.1%}"],
            textposition="outside"
        ))
        fig_ps.update_layout(
            title="Shootout Win Probability Comparison",
            yaxis=dict(tickformat=".0%", range=[0, 1.1]),
            height=280,
            showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_ps, width="stretch")

        st.info(
            f"Average rounds to resolve shootout: **{ps_result['avg_rounds']:.1f}** rounds. "
            f"Note: Taker ratings assume baseline skill. Argentina (Emi Martínez) / Croatia (Livaković) keepers "
            f"provide a significant penalty-saving advantage."
        )

else:
    st.info("Configure teams in the sidebar and click **Predict Matchup**.")
