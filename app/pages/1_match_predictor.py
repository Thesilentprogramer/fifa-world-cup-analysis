"""WC 2026 lean dashboard — today, tomorrow, played (cached).

Phase 5 additions:
  - 🔄 Refresh Live Data button (martj42 + fixtures + rebuild features + conditional retrain)
  - Model vs Market accuracy panel in Model Info expander
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api_football_client import get_fixture_sync_meta, load_wc_fixtures, sync_wc_fixtures
from src.fixture_filters import is_sync_stale, split_today_tomorrow
from src.match_predictor import get_predictor
from src.prematch_analysis import analyze_fixture, load_played_cached, load_upcoming_today_tomorrow
from app.theme import inject_app_styles
from app.charts.model_charts import fig_match_model_performance, load_match_metrics

inject_app_styles()

st.title("WC 2026 Dashboard")
st.caption("Today & tomorrow fixtures · cached played results")


@st.cache_resource
def load_predictor():
    return get_predictor()


try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(f"Model not found. Run `python scripts/train_model.py` first.\n\n{e}")
    st.stop()

teams = sorted(predictor.teams)


# ---------------------------------------------------------------------------
# 🔄 Live Data Refresh
# ---------------------------------------------------------------------------

def _run_full_refresh() -> dict:
    """Run the full daily refresh pipeline: download → features → conditional retrain."""
    log: list[str] = []
    result: dict = {"success": True, "log": log, "new_finished": 0, "retrained": False}
    try:
        from scripts.download_data import download_international
        from src.api_football_client import sync_wc_fixtures
        from src.feature_engineering import build_features
        import json, datetime as dt

        log.append("📥 Downloading martj42 international results...")
        download_international(force=True)
        log.append("✅ Results downloaded")

        log.append("📡 Syncing WC 2026 fixtures...")
        fixtures = sync_wc_fixtures(force=True, known_teams=teams)
        finished_count = int((fixtures["is_finished"] == True).sum()) if not fixtures.empty else 0
        log.append(f"✅ {len(fixtures)} fixtures synced ({finished_count} finished)")

        log.append("⚙️  Rebuilding feature table...")
        build_features()
        log.append("✅ Features rebuilt")

        # Check if new finished matches warrant retrain
        LAST_REFRESH = PROJECT_ROOT / "data" / "processed" / "last_refresh.json"
        prev_finished = 0
        if LAST_REFRESH.exists():
            try:
                meta = json.loads(LAST_REFRESH.read_text())
                prev_finished = int(meta.get("finished_count", 0))
            except Exception:
                pass

        new_finished = finished_count - prev_finished
        result["new_finished"] = new_finished

        if new_finished > 0:
            log.append(f"🧠 {new_finished} new finished match(es) → retraining model...")
            from scripts.train_model import main as train_main
            train_main()
            result["retrained"] = True
            log.append("✅ Model retrained")
        else:
            log.append("ℹ️  No new finished matches — skipping retrain")

        # Save refresh metadata
        meta_out = {
            "refreshed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "finished_count": finished_count,
            "new_finished": new_finished,
            "retrained": result["retrained"],
        }
        LAST_REFRESH.parent.mkdir(parents=True, exist_ok=True)
        LAST_REFRESH.write_text(json.dumps(meta_out, indent=2))

    except Exception as ex:
        result["success"] = False
        log.append(f"❌ Error: {ex}")

    return result


# ---------------------------------------------------------------------------
# Refresh button (sidebar)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔄 Live Data Refresh")
    st.caption("Pulls latest results from martj42 GitHub repo (updated daily)")
    refresh_btn = st.button("Refresh Live Data", type="primary", use_container_width=True)
    last_meta = get_fixture_sync_meta()
    if last_meta:
        ts = last_meta.get("synced_at", "never")[:19]
        src = last_meta.get("source", "?")
        st.caption(f"Last sync: {ts} · {src}")

if refresh_btn:
    st.session_state["refresh_running"] = True
    progress_box = st.empty()
    with progress_box.container():
        st.info("🔄 Running data refresh pipeline…")
        log_area = st.empty()
        with st.spinner("Downloading & rebuilding…"):
            res = _run_full_refresh()
        log_area.code("\n".join(res["log"]))

    if res["success"]:
        msg = f"✅ Refresh complete! {res['new_finished']} new match(es)"
        if res["retrained"]:
            msg += " · Model retrained 🎯"
        st.toast(msg, icon="✅")
        # Reload predictor with fresh model
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.fixtures_synced_session = True
        st.rerun()
    else:
        st.toast("⚠️ Refresh failed — see log above", icon="⚠️")


# ---------------------------------------------------------------------------
# SHAP / WDL / card helpers
# ---------------------------------------------------------------------------

def _render_shap_panel(result: dict, team_a: str, team_b: str) -> None:
    st.markdown(result.get("narrative", ""))
    shap_data = result.get("shap", [])
    if not shap_data:
        return
    features = [d["label"] for d in shap_data][::-1]
    values = [d["shap_value"] for d in shap_data][::-1]
    colors_bar = ["#2E7D32" if v > 0 else "#C62828" for v in values]
    fig_shap = go.Figure(go.Bar(
        x=values, y=features, orientation="h", marker_color=colors_bar,
    ))
    fig_shap.update_layout(
        title="Top 5 SHAP Features",
        xaxis_title="SHAP value",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_shap, width="stretch")


def _render_wdl_donut(probs: dict, team_a: str, team_b: str) -> None:
    labels = [f"{team_a} Win", "Draw", f"{team_b} Win"]
    values = [probs["win"], probs["draw"], probs["loss"]]
    colors = ["#2E7D32", "#F9A825", "#C62828"]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.45,
        marker=dict(colors=colors), textinfo="label+percent",
        pull=[0.05 if v == max(values) else 0 for v in values],
    )])
    fig.update_layout(showlegend=False, height=360, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, width="stretch")


def _format_kickoff(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    local = t.tz_convert(datetime.now().astimezone().tzinfo)
    return local.strftime("%H:%M")


def _render_market_comparison(home: str, away: str, model_probs: dict) -> None:
    snap_a = predictor._team_snapshots.get(home)
    
    poly_win = snap_a.get("polymarket_prob_win") if snap_a is not None else None
    poly_draw = snap_a.get("polymarket_prob_draw") if snap_a is not None else None
    poly_loss = snap_a.get("polymarket_prob_loss") if snap_a is not None else None
    
    bk_win = snap_a.get("implied_prob_win") if snap_a is not None else None
    bk_draw = snap_a.get("implied_prob_draw") if snap_a is not None else None
    bk_loss = snap_a.get("implied_prob_loss") if snap_a is not None else None

    rows = [{
        "Source": "🤖 Our Model",
        f"{home} Win": f"{model_probs['win']:.1%}",
        "Draw": f"{model_probs['draw']:.1%}",
        f"{away} Win": f"{model_probs['away']:.1%}",
    }]

    if poly_win and pd.notna(poly_win) and float(poly_win) > 0:
        rows.append({
            "Source": "📊 Polymarket",
            f"{home} Win": f"{float(poly_win):.1%}",
            "Draw": f"{float(poly_draw):.1%}" if poly_draw and pd.notna(poly_draw) else "N/A",
            f"{away} Win": f"{float(poly_loss):.1%}" if poly_loss and pd.notna(poly_loss) else "N/A",
        })

    if bk_win and pd.notna(bk_win) and float(bk_win) > 0:
        rows.append({
            "Source": "🏦 Bookmaker",
            f"{home} Win": f"{float(bk_win):.1%}",
            "Draw": f"{float(bk_draw):.1%}" if bk_draw and pd.notna(bk_draw) else "N/A",
            f"{away} Win": f"{float(bk_loss):.1%}" if bk_loss and pd.notna(bk_loss) else "N/A",
        })

    st.markdown("##### 📈 Model vs Market Odds")
    comparison_df = pd.DataFrame(rows)
    st.table(comparison_df.set_index("Source"))

    if len(rows) >= 2:
        sources = [r["Source"] for r in rows]
        win_vals = []
        for r in rows:
            v = r[f"{home} Win"].replace("%", "")
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
            title=f"{home} Win Probability — Model vs Market",
            yaxis=dict(tickformat=".0%", range=[0, 1.1]),
            height=260,
            showlegend=False,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_comp, width="stretch")


def _match_card(row: pd.Series, key_prefix: str) -> None:
    home, away = row["home_team"], row["away_team"]
    kickoff = _format_kickoff(row["match_date"])
    pred = row.get("prediction", "—")
    conf = row.get("confidence")
    conf_s = f"{conf:.0%}" if pd.notna(conf) else "—"

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            st.markdown(f"**{home}** vs **{away}**")
            st.caption(f"{kickoff} · {row.get('venue', 'TBD')}")
        with c2:
            st.write(f"Pick: **{pred}**")
            if pd.notna(row.get("prob_home")):
                st.caption(
                    f"P(H/D/A): {row['prob_home']:.0%} / {row['prob_draw']:.0%} / {row['prob_away']:.0%}"
                )
        with c3:
            st.metric("Conf.", conf_s)

        detail_key = f"{key_prefix}_detail_{row.get('_fixture_id', home)}"
        if st.checkbox("Deep dive", key=detail_key):
            analysis = analyze_fixture(
                home, away,
                stage=row.get("stage", "group"),
                predictor=predictor,
                n_xg_sims=2000,
                include_shap=True,
            )
            col_a, col_b = st.columns(2)
            with col_a:
                _render_wdl_donut({
                    "win": analysis["prob_home_win"],
                    "draw": analysis["prob_draw"],
                    "away": analysis["prob_away_win"],
                }, home, away)
            with col_b:
                if analysis["expected_xg_home"] is not None:
                    st.metric(home, f"{analysis['expected_xg_home']:.2f} xG")
                    st.metric(away, f"{analysis['expected_xg_away']:.2f} xG")
                    if analysis.get("top_scoreline"):
                        st.caption(f"Likeliest: {analysis['top_scoreline']}")
                _render_shap_panel(analysis, home, away)
            
            # Show side-by-side market benchmark
            _render_market_comparison(home, away, {
                "win": analysis["prob_home_win"],
                "draw": analysis["prob_draw"],
                "away": analysis["prob_away_win"]
            })


def _played_card(row: pd.Series) -> None:
    home, away = row["home_team"], row["away_team"]
    pred = row.get("prediction") or row.get("prediction_label", "—")
    hs, aws = row.get("home_score"), row.get("away_score")
    score_s = f"{int(hs)}–{int(aws)}" if pd.notna(hs) and pd.notna(aws) else "—"

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            st.markdown(f"**{home}** vs **{away}**")
            st.caption(pd.Timestamp(row["match_date"]).strftime("%Y-%m-%d %H:%M"))
        with c2:
            st.write(f"Predicted: **{pred}** · Final: **{score_s}**")
            if row.get("pred_correct") is True:
                st.success("Correct")
            elif row.get("pred_correct") is False:
                st.error("Wrong")
        with c3:
            conf = row.get("confidence")
            st.metric("Conf.", f"{conf:.0%}" if pd.notna(conf) else "—")


# ---------------------------------------------------------------------------
# Auto-sync if stale (once per session)
# ---------------------------------------------------------------------------
meta = get_fixture_sync_meta()
if "fixtures_synced_session" not in st.session_state:
    st.session_state.fixtures_synced_session = False

if not st.session_state.fixtures_synced_session and is_sync_stale(meta.get("synced_at") if meta else None):
    with st.spinner("Refreshing fixture cache..."):
        sync_wc_fixtures(force=False, known_teams=teams)
    st.session_state.fixtures_synced_session = True
    st.cache_data.clear()

col_sync, col_info = st.columns([1, 3])
with col_sync:
    if st.button("Refresh fixtures"):
        with st.spinner("Syncing..."):
            sync_wc_fixtures(force=True, known_teams=teams)
        st.session_state.fixtures_synced_session = True
        st.cache_data.clear()
        st.rerun()

fixtures = load_wc_fixtures()
if fixtures.empty:
    st.warning("No WC 2026 fixtures. Add `API_FOOTBALL_KEY` to `.env` or click Refresh.")
    st.stop()

meta = get_fixture_sync_meta()
sync_line = (
    f"Last sync: {meta.get('synced_at', 'never')[:19]} · source: {meta.get('source', '?')}"
    if meta else "No sync metadata"
)
with col_info:
    st.caption(sync_line)

if meta and meta.get("api_error"):
    st.warning(
        f"API-Football: {meta['api_error']} — using **martj42** fixture cache. "
        "Free plans may not include the 2026 season; upgrade at api-football.com for live WC 2026 schedules."
    )

today_raw, tomorrow_raw = split_today_tomorrow(fixtures)
n_show = len(today_raw) + len(tomorrow_raw)
st.caption(f"Showing **{n_show}** upcoming matches (today + tomorrow)")


@st.cache_data(ttl=900, show_spinner=False)
def _load_upcoming_cached():
    p = load_predictor()
    return load_upcoming_today_tomorrow(predictor=p, n_xg_sims=500)


@st.cache_data(ttl=900, show_spinner=False)
def _load_played_cached(_fixtures_version: str):
    p = load_predictor()
    return load_played_cached(fixtures, predictor=p)


with st.spinner("Analyzing upcoming matches..."):
    today_df, tomorrow_df = _load_upcoming_cached()

st.subheader("Today")
if today_df.empty:
    st.info("No matches scheduled for today.")
else:
    for _, row in today_df.iterrows():
        _match_card(row, "today")

st.subheader("Tomorrow")
if tomorrow_df.empty:
    st.info("No matches scheduled for tomorrow.")
else:
    for _, row in tomorrow_df.iterrows():
        _match_card(row, "tomorrow")

st.subheader("Played — cached predictions")
played_df = _load_played_cached(str(meta.get("synced_at", "")) if meta else "none")
if played_df.empty:
    st.info("No played matches in cache yet.")
else:
    st.caption(f"{len(played_df)} finished matches · predictions frozen at kickoff")
    for _, row in played_df.head(20).iterrows():
        _played_card(row)

# ---------------------------------------------------------------------------
# Model info + Market Benchmark (Phase 5)
# ---------------------------------------------------------------------------

with st.expander("📊 Model info & Market Benchmark"):
    metrics = load_match_metrics()
    m1, m2, m3 = st.columns(3)
    m1.metric("Val (2018 WC)", f"{metrics.get('validation', {}).get('accuracy', 0):.1%}")
    m2.metric("Test (2022 WC)", f"{metrics.get('test', {}).get('accuracy', 0):.1%}")
    m3.metric("High-conf test", f"{metrics.get('high_confidence_test_acc', 0):.1%}")
    st.plotly_chart(fig_match_model_performance(), width="stretch")

    # Phase 5: Model vs Market accuracy table
    st.markdown("#### Model vs Market")
    st.caption(
        "Polymarket/bookmaker implied probabilities convert to implied picks. "
        "Comparison is directional — market odds not always available for all historical WC matches."
    )
    comp_data = {
        "Metric": ["Accuracy", "Log-Loss", "Brier Score"],
        "Model (test 2022 WC)": [
            f"{metrics.get('test', {}).get('accuracy', 0):.1%}",
            f"{metrics.get('test', {}).get('log_loss', 0):.3f}",
            f"{metrics.get('test', {}).get('brier', 0):.3f}",
        ],
        "Bookmaker baseline": ["~58–62%", "~0.95", "~0.18"],
        "Random baseline": ["~33%", "~1.10", "~0.22"],
    }
    st.table(comp_data)
    st.caption("Custom head-to-head picks: **Custom Matchup** in the sidebar.")

st.caption("Fixtures: API-Football or martj42 fallback. WC 2026 is out-of-sample.")
