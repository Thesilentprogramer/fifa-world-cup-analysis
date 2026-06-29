"""Pre-match analysis bundle: W/D/L model, xG simulation, SHAP."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.api_football_client import load_wc_fixtures
from src.fixture_filters import filter_played, filter_upcoming_window, split_today_tomorrow
from src.match_predictor import MatchPredictor, get_predictor
from src.prediction_cache import log_prediction_from_analysis, merge_played_with_log, update_played_results
from src.statsbomb_shots import load_statsbomb_shots
from src.xg_engine import load_xg_model, simulate_match


def _actual_result_home(home_score: Any, away_score: Any) -> str | None:
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    hs, aws = int(home_score), int(away_score)
    if hs > aws:
        return "Home Win"
    if hs < aws:
        return "Away Win"
    return "Draw"


def _prediction_correct(pred_home_perspective: str, actual: str | None) -> bool | None:
    """pred is Win/Loss/Draw from home team perspective."""
    if actual is None:
        return None
    mapping = {"Win": "Home Win", "Loss": "Away Win", "Draw": "Draw"}
    return mapping.get(pred_home_perspective) == actual


def analyze_fixture(
    home_team: str,
    away_team: str,
    stage: str = "group",
    predictor: MatchPredictor | None = None,
    n_xg_sims: int = 3000,
    include_shap: bool = True,
) -> dict:
    """Full pre-match analysis for one fixture (home team perspective, neutral site)."""
    predictor = predictor or get_predictor()

    if include_shap:
        wdl = predictor.predict(home_team, away_team, stage=stage, is_home=0)
    else:
        wdl = predictor.predict_fast(home_team, away_team, stage=stage, is_home=0)

    xg_result = None
    xg_error = None
    try:
        shots = load_statsbomb_shots()
        model = load_xg_model()
        xg_result = simulate_match(
            home_team, away_team,
            n_simulations=n_xg_sims,
            shots=shots,
            model=model,
        )
    except (FileNotFoundError, ImportError, OSError) as e:
        xg_error = str(e)

    snap_home = predictor._team_snapshots.get(home_team)
    snap_away = predictor._team_snapshots.get(away_team)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "stage": stage,
        "prob_home_win": wdl["probabilities"]["win"],
        "prob_draw": wdl["probabilities"]["draw"],
        "prob_away_win": wdl["probabilities"]["loss"],
        "predicted_outcome": wdl["predicted_outcome"],
        "confidence": wdl["confidence"],
        "shap": wdl.get("shap", []),
        "narrative": wdl.get("narrative", ""),
        "sparse_data_warning": wdl["sparse_data_warning"],
        "expected_xg_home": xg_result["expected_xg_a"] if xg_result else None,
        "expected_xg_away": xg_result["expected_xg_b"] if xg_result else None,
        "xg_prob_home": xg_result["prob_win_a"] if xg_result else None,
        "xg_prob_draw": xg_result["prob_draw"] if xg_result else None,
        "xg_prob_away": xg_result["prob_win_b"] if xg_result else None,
        "top_scoreline": (
            f"{xg_result['top_scorelines'][0]['score_a']}-{xg_result['top_scorelines'][0]['score_b']}"
            if xg_result and xg_result.get("top_scorelines") else None
        ),
        "top_scoreline_prob": (
            xg_result["top_scorelines"][0]["probability"]
            if xg_result and xg_result.get("top_scorelines") else None
        ),
        "elo_home": float(snap_home["team_elo"]) if snap_home is not None and "team_elo" in snap_home else None,
        "elo_away": float(snap_away["team_elo"]) if snap_away is not None and "team_elo" in snap_away else None,
        "xg_error": xg_error,
    }


def analyze_fixtures(
    fixtures: pd.DataFrame,
    predictor: MatchPredictor | None = None,
    n_xg_sims: int = 1000,
    include_shap: bool = False,
) -> pd.DataFrame:
    """Run pre-match analysis for all rows in a fixtures table."""
    predictor = predictor or get_predictor()
    records = []

    shots = None
    xg_model = None
    try:
        shots = load_statsbomb_shots()
        xg_model = load_xg_model()
    except (FileNotFoundError, ImportError, OSError):
        pass

    for _, fix in fixtures.iterrows():
        if include_shap:
            wdl_part = analyze_fixture(
                fix["home_team"], fix["away_team"],
                stage=fix.get("stage", "group"),
                predictor=predictor, n_xg_sims=0, include_shap=True,
            )
        else:
            wdl = predictor.predict_fast(
                fix["home_team"], fix["away_team"],
                stage=fix.get("stage", "group"), is_home=0,
            )
            wdl_part = {
                "prob_home_win": wdl["probabilities"]["win"],
                "prob_draw": wdl["probabilities"]["draw"],
                "prob_away_win": wdl["probabilities"]["loss"],
                "predicted_outcome": wdl["predicted_outcome"],
                "confidence": wdl["confidence"],
                "shap": [], "narrative": "", "sparse_data_warning": wdl["sparse_data_warning"],
                "expected_xg_home": None, "expected_xg_away": None,
                "xg_prob_home": None, "xg_prob_draw": None, "xg_prob_away": None,
                "top_scoreline": None, "top_scoreline_prob": None,
                "elo_home": None, "elo_away": None, "xg_error": None,
            }
            snap_home = predictor._team_snapshots.get(fix["home_team"])
            snap_away = predictor._team_snapshots.get(fix["away_team"])
            if snap_home is not None and "team_elo" in snap_home:
                wdl_part["elo_home"] = float(snap_home["team_elo"])
            if snap_away is not None and "team_elo" in snap_away:
                wdl_part["elo_away"] = float(snap_away["team_elo"])

            if shots is not None and xg_model is not None and not shots.empty:
                xg_result = simulate_match(
                    fix["home_team"], fix["away_team"],
                    n_simulations=n_xg_sims, shots=shots, model=xg_model,
                )
                wdl_part.update({
                    "expected_xg_home": xg_result["expected_xg_a"],
                    "expected_xg_away": xg_result["expected_xg_b"],
                    "xg_prob_home": xg_result["prob_win_a"],
                    "xg_prob_draw": xg_result["prob_draw"],
                    "xg_prob_away": xg_result["prob_win_b"],
                    "top_scoreline": (
                        f"{xg_result['top_scorelines'][0]['score_a']}-{xg_result['top_scorelines'][0]['score_b']}"
                        if xg_result.get("top_scorelines") else None
                    ),
                    "top_scoreline_prob": (
                        xg_result["top_scorelines"][0]["probability"]
                        if xg_result.get("top_scorelines") else None
                    ),
                })

        analysis = wdl_part
        actual = _actual_result_home(fix.get("home_score"), fix.get("away_score"))
        pred_label = analysis["predicted_outcome"]
        if pred_label == "Win":
            pick = f"{fix['home_team']} Win"
        elif pred_label == "Loss":
            pick = f"{fix['away_team']} Win"
        else:
            pick = "Draw"

        records.append({
            "fixture_id": fix.get("fixture_id"),
            "match_date": fix["match_date"],
            "home_team": fix["home_team"],
            "away_team": fix["away_team"],
            "stage": fix.get("stage", "group"),
            "round": fix.get("round", ""),
            "venue": fix.get("venue", ""),
            "status": fix.get("status", ""),
            "is_finished": fix.get("is_finished", False),
            "is_upcoming": fix.get("is_upcoming", True),
            "home_score": fix.get("home_score"),
            "away_score": fix.get("away_score"),
            "actual_result": actual,
            "prediction": pick,
            "pred_correct": _prediction_correct(pred_label, actual),
            "prob_home": analysis["prob_home_win"],
            "prob_draw": analysis["prob_draw"],
            "prob_away": analysis["prob_away_win"],
            "confidence": analysis["confidence"],
            "expected_xg_home": analysis["expected_xg_home"],
            "expected_xg_away": analysis["expected_xg_away"],
            "top_scoreline": analysis["top_scoreline"],
            "elo_home": analysis["elo_home"],
            "elo_away": analysis["elo_away"],
            "_analysis": analysis,
            "_fixture_id": f"{fix['home_team']}|{fix['away_team']}|{fix['match_date']}",
        })

    return pd.DataFrame(records)


def analyze_fixtures_subset(
    fixtures: pd.DataFrame,
    predictor: MatchPredictor | None = None,
    n_xg_sims: int = 1000,
    include_shap: bool = False,
    cache_predictions: bool = False,
) -> pd.DataFrame:
    """Run analysis on a pre-filtered fixtures frame; optionally log kickoff predictions."""
    df = analyze_fixtures(
        fixtures,
        predictor=predictor,
        n_xg_sims=n_xg_sims,
        include_shap=include_shap,
    )
    if cache_predictions and not df.empty:
        for _, row in df.iterrows():
            analysis = row.get("_analysis") or {}
            if not analysis:
                continue
            log_prediction_from_analysis(row, analysis)
    return df


def load_played_cached(
    fixtures: pd.DataFrame | None = None,
    predictor: MatchPredictor | None = None,
) -> pd.DataFrame:
    """Played fixtures with cached prediction log (backfills missing entries)."""
    fixtures = fixtures if fixtures is not None else load_wc_fixtures()
    if fixtures.empty:
        return pd.DataFrame()
    predictor = predictor or get_predictor()
    return merge_played_with_log(fixtures, predictor=predictor)


def load_upcoming_today_tomorrow(
    predictor: MatchPredictor | None = None,
    n_xg_sims: int = 800,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze only today and tomorrow upcoming fixtures."""
    fixtures = load_wc_fixtures()
    if fixtures.empty:
        return pd.DataFrame(), pd.DataFrame()
    today_f, tomorrow_f = split_today_tomorrow(fixtures)
    today_a = analyze_fixtures_subset(
        today_f, predictor=predictor, n_xg_sims=n_xg_sims,
        include_shap=False, cache_predictions=True,
    )
    tomorrow_a = analyze_fixtures_subset(
        tomorrow_f, predictor=predictor, n_xg_sims=n_xg_sims,
        include_shap=False, cache_predictions=True,
    )
    return today_a, tomorrow_a


def load_and_analyze_wc2026(
    filter_status: str = "all",
    predictor: MatchPredictor | None = None,
    n_xg_sims: int = 3000,
) -> pd.DataFrame:
    """Load cached WC 2026 fixtures and run analysis."""
    fixtures = load_wc_fixtures()
    if fixtures.empty:
        return pd.DataFrame()

    if filter_status == "upcoming":
        fixtures = filter_upcoming_window(fixtures)
    elif filter_status == "played":
        fixtures = filter_played(fixtures)
        return load_played_cached(fixtures)

    return analyze_fixtures_subset(fixtures, predictor=predictor, n_xg_sims=n_xg_sims)
