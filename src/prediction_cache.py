"""Kickoff prediction log for played-match review."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTION_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "prediction_log.parquet"

LOG_COLUMNS = [
    "fixture_id",
    "match_date",
    "home_team",
    "away_team",
    "predicted_outcome",
    "prediction_label",
    "prob_home",
    "prob_draw",
    "prob_away",
    "confidence",
    "logged_at",
    "home_score",
    "away_score",
    "pred_correct",
]


def _fixture_key(row: pd.Series) -> str:
    fid = row.get("fixture_id")
    if pd.notna(fid) and str(fid).strip():
        return str(fid)
    md = pd.Timestamp(row["match_date"]).strftime("%Y%m%d%H%M")
    return f"{md}_{row['home_team']}_{row['away_team']}"


def load_prediction_log() -> pd.DataFrame:
    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_parquet(PREDICTION_LOG_PATH)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def save_prediction_log(df: pd.DataFrame) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = df[LOG_COLUMNS].copy()
    out.to_parquet(PREDICTION_LOG_PATH, index=False)


def _outcome_label(pred: str, home: str, away: str) -> str:
    if pred == "Win":
        return f"{home} Win"
    if pred == "Loss":
        return f"{away} Win"
    return "Draw"


def _actual_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "Home Win"
    if home_score < away_score:
        return "Away Win"
    return "Draw"


def _pred_correct(pred: str, home_score: int, away_score: int) -> bool:
    actual = _actual_result(home_score, away_score)
    mapping = {"Win": "Home Win", "Loss": "Away Win", "Draw": "Draw"}
    return mapping.get(pred) == actual


def log_prediction_from_analysis(row: pd.Series, analysis: dict) -> None:
    """Append or skip if fixture already logged."""
    log = load_prediction_log()
    key = _fixture_key(row)
    if not log.empty and "fixture_id" in log.columns:
        existing = log["fixture_id"].astype(str)
        if key in existing.values:
            return

    pred = analysis.get("predicted_outcome", "Draw")
    hs, aws = row.get("home_score"), row.get("away_score")
    pred_correct = None
    if pd.notna(hs) and pd.notna(aws) and pd.notna(pred):
        pred_correct = _pred_correct(str(pred), int(hs), int(aws))

    entry = {
        "fixture_id": key,
        "match_date": pd.Timestamp(row["match_date"]),
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "predicted_outcome": pred,
        "prediction_label": analysis.get("prediction_label") or _outcome_label(
            pred, row["home_team"], row["away_team"]
        ),
        "prob_home": analysis.get("prob_home_win") or analysis.get("prob_home"),
        "prob_draw": analysis.get("prob_draw"),
        "prob_away": analysis.get("prob_away_win") or analysis.get("prob_away"),
        "confidence": analysis.get("confidence"),
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "home_score": hs,
        "away_score": aws,
        "pred_correct": pred_correct,
    }
    log = pd.concat([log, pd.DataFrame([entry])], ignore_index=True)
    save_prediction_log(log)


def _analysis_from_predict(wdl: dict, home: str, away: str) -> dict:
    pred = wdl["predicted_outcome"]
    return {
        "predicted_outcome": pred,
        "prob_home_win": wdl["probabilities"]["win"],
        "prob_draw": wdl["probabilities"]["draw"],
        "prob_away_win": wdl["probabilities"]["loss"],
        "confidence": wdl["confidence"],
        "prediction_label": _outcome_label(pred, home, away),
    }


def ensure_played_predictions(
    fixtures: pd.DataFrame,
    predictor=None,
) -> int:
    """Log model predictions for finished fixtures missing from prediction_log."""
    if fixtures.empty:
        return 0

    played = fixtures[fixtures["is_finished"] == True].copy()  # noqa: E712
    if played.empty:
        return 0

    if predictor is None:
        from src.match_predictor import get_predictor
        predictor = get_predictor()

    log = load_prediction_log()
    existing = set(log["fixture_id"].astype(str)) if not log.empty else set()
    added = 0

    for _, fix in played.iterrows():
        key = _fixture_key(fix)
        if key in existing:
            continue
        wdl = predictor.predict_fast(
            fix["home_team"],
            fix["away_team"],
            stage=fix.get("stage", "group"),
            is_home=0,
        )
        analysis = _analysis_from_predict(wdl, fix["home_team"], fix["away_team"])
        log_prediction_from_analysis(fix, {
            **analysis,
            "prob_home_win": analysis["prob_home_win"],
            "prob_away_win": analysis["prob_away_win"],
        })
        existing.add(key)
        added += 1

    if added:
        update_played_results(fixtures)
    return added


def update_played_results(fixtures: pd.DataFrame) -> int:
    """Backfill scores and pred_correct for finished matches. Returns rows updated."""
    log = load_prediction_log()
    if log.empty or fixtures.empty:
        return 0

    played = fixtures[fixtures["is_finished"] == True].copy()  # noqa: E712
    if played.empty:
        return 0

    updated = 0
    for _, fix in played.iterrows():
        key = _fixture_key(fix)
        mask = log["fixture_id"].astype(str) == key
        if not mask.any():
            continue
        hs, aws = fix.get("home_score"), fix.get("away_score")
        if pd.isna(hs) or pd.isna(aws):
            continue
        hs, aws = int(hs), int(aws)
        idx = log.index[mask][0]
        if log.at[idx, "home_score"] != hs or log.at[idx, "away_score"] != aws:
            log.at[idx, "home_score"] = hs
            log.at[idx, "away_score"] = aws
            pred = log.at[idx, "predicted_outcome"]
            if pd.notna(pred):
                log.at[idx, "pred_correct"] = _pred_correct(str(pred), hs, aws)
            updated += 1
        elif pd.isna(log.at[idx, "pred_correct"]) and pd.notna(log.at[idx, "predicted_outcome"]):
            log.at[idx, "pred_correct"] = _pred_correct(str(log.at[idx, "predicted_outcome"]), hs, aws)
            updated += 1

    if updated:
        save_prediction_log(log)
    return updated


def merge_played_with_log(
    fixtures: pd.DataFrame,
    analyzed: pd.DataFrame | None = None,
    predictor=None,
) -> pd.DataFrame:
    """Played fixtures enriched with cached predictions (backfills if missing)."""
    played = fixtures[fixtures["is_finished"] == True].copy()  # noqa: E712
    if played.empty:
        return played

    ensure_played_predictions(fixtures, predictor=predictor)
    update_played_results(fixtures)

    log = load_prediction_log()
    played["fixture_id"] = played.apply(_fixture_key, axis=1)
    played["match_date"] = pd.to_datetime(played["match_date"], utc=True, errors="coerce")

    if log.empty:
        if analyzed is not None and not analyzed.empty:
            return analyzed[analyzed["is_finished"] == True].copy()  # noqa: E712
        return played

    log = log.copy()
    log["fixture_id"] = log["fixture_id"].astype(str)
    pred_cols = [
        "fixture_id", "prediction_label", "prob_home", "prob_draw",
        "prob_away", "confidence", "pred_correct", "predicted_outcome",
    ]
    log_slim = log[[c for c in pred_cols if c in log.columns]].drop_duplicates("fixture_id")

    merged = played.merge(log_slim, on="fixture_id", how="left")
    merged["prediction"] = merged["prediction_label"]

    return merged.sort_values("match_date", ascending=False).reset_index(drop=True)
