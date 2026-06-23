"""Plotly chart helpers for model showcase sections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_match_metrics() -> dict:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_xg_metrics() -> dict:
    path = MODELS_DIR / "xg_model_meta.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("metrics", {})


def fig_match_model_performance() -> go.Figure:
    m = load_match_metrics()
    if not m:
        return go.Figure().add_annotation(text="No metrics.json found", showarrow=False)

    splits = ["validation", "test"]
    labels = ["2018 WC (val)", "2022 WC (test)"]
    acc = [m.get(s, {}).get("accuracy", 0) * 100 for s in splits]
    logloss = [m.get(s, {}).get("log_loss", 0) for s in splits]
    brier = [m.get(s, {}).get("brier", 0) for s in splits]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(name="Accuracy %", x=labels, y=acc, marker_color=["#2E7D32", "#1565C0"]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(name="Log-loss", x=labels, y=logloss, mode="lines+markers", line=dict(color="#F9A825")),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(name="Brier", x=labels, y=brier, mode="lines+markers", line=dict(color="#C62828")),
        secondary_y=True,
    )
    fig.update_layout(
        title="Match Outcome Model — Validation vs Test",
        yaxis_title="Accuracy %",
        yaxis2_title="Log-loss / Brier",
        height=360,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(range=[0, 100], secondary_y=False)
    return fig


def fig_wc_prediction_breakdown(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty or "prediction" not in df.columns:
        return go.Figure().add_annotation(text="No fixture predictions yet", showarrow=False)

    counts = df["prediction"].value_counts().reset_index()
    counts.columns = ["prediction", "count"]
    fig = px.pie(
        counts, names="prediction", values="count",
        title="WC 2026 — Predicted Outcomes",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=340, margin=dict(t=50, b=20))
    return fig


def fig_confidence_histogram(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty or "confidence" not in df.columns:
        return go.Figure().add_annotation(text="No confidence data", showarrow=False)

    fig = px.histogram(
        df, x="confidence", nbins=20,
        title="Prediction Confidence Distribution",
        labels={"confidence": "Confidence", "count": "Matches"},
        color_discrete_sequence=["#1565C0"],
    )
    fig.add_vline(x=0.55, line_dash="dash", line_color="#F9A825", annotation_text="High-conf threshold")
    fig.update_layout(height=320, margin=dict(t=50, b=40), bargap=0.05)
    return fig


def fig_xg_calibration(shots: pd.DataFrame, model) -> go.Figure:
    """Bin predicted xG vs actual goal rate."""
    if shots is None or shots.empty or model is None:
        return go.Figure().add_annotation(text="No shot data for calibration", showarrow=False)

    from src.xg_engine import prepare_shot_features

    X, y = prepare_shot_features(shots)
    if X.empty:
        return go.Figure().add_annotation(text="No valid shots", showarrow=False)

    probs = model.predict_proba(X)[:, 1]
    bins = pd.qcut(probs, q=10, duplicates="drop")
    cal = (
        pd.DataFrame({"pred": probs, "goal": y.values, "bin": bins})
        .groupby("bin", observed=True)
        .agg(mean_pred=("pred", "mean"), goal_rate=("goal", "mean"), n=("goal", "count"))
        .reset_index(drop=True)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cal["mean_pred"], y=cal["goal_rate"],
        mode="lines+markers", name="Model",
        line=dict(color="#2E7D32", width=2),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0.5], y=[0, 0.5],
        mode="lines", name="Perfect calibration",
        line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        title="xG Model Calibration (decile bins)",
        xaxis_title="Mean predicted xG",
        yaxis_title="Actual goal rate",
        height=340,
        margin=dict(t=50, b=40),
    )
    return fig


def fig_xg_distance_curve(shots: pd.DataFrame, model) -> go.Figure:
    if shots is None or shots.empty or model is None:
        return go.Figure().add_annotation(text="No shot data", showarrow=False)

    from src.xg_engine import prepare_shot_features

    X, y = prepare_shot_features(shots)
    if X.empty:
        return go.Figure().add_annotation(text="No valid shots", showarrow=False)

    probs = model.predict_proba(X)[:, 1]
    df = X.copy()
    df["pred_xg"] = probs
    df["goal"] = y.values
    df["dist_bin"] = pd.cut(df["distance"], bins=12)
    curve = (
        df.groupby("dist_bin", observed=True)
        .agg(mean_dist=("distance", "mean"), mean_xg=("pred_xg", "mean"), goal_rate=("goal", "mean"))
        .reset_index(drop=True)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["mean_dist"], y=curve["mean_xg"],
        mode="lines+markers", name="Predicted xG",
        line=dict(color="#1565C0"),
    ))
    fig.add_trace(go.Scatter(
        x=curve["mean_dist"], y=curve["goal_rate"],
        mode="lines+markers", name="Actual goal rate",
        line=dict(color="#C62828", dash="dot"),
    ))
    fig.update_layout(
        title="Distance vs xG / Goal Rate",
        xaxis_title="Shot distance (yards)",
        yaxis_title="Rate",
        height=340,
        margin=dict(t=50, b=40),
    )
    return fig


def fig_xg_scoreline_heatmap(result: dict) -> go.Figure:
    top = result.get("top_scorelines", [])[:15]
    if not top:
        return go.Figure().add_annotation(text="Run simulation for scorelines", showarrow=False)

    scores_a = [int(s["score_a"]) for s in top]
    scores_b = [int(s["score_b"]) for s in top]
    probs = [s["probability"] for s in top]
    labels = [f"{a}-{b}" for a, b in zip(scores_a, scores_b)]

    fig = go.Figure(go.Bar(x=labels, y=probs, marker_color="#2E7D32"))
    fig.update_layout(
        title="Top Scoreline Probabilities",
        yaxis_tickformat=".1%",
        height=320,
        margin=dict(t=50, b=40),
    )
    return fig
