"""Match outcome prediction with SHAP explanations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.elo import INITIAL_ELO
from src.feature_engineering import FEATURE_COLUMNS, build_features, get_latest_team_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "match_features.parquet"

FEATURE_LABELS = {
    "is_home": "home advantage",
    "is_neutral": "neutral venue",
    "is_knockout": "knockout stage pressure",
    "same_confederation": "same confederation",
    "team_elo": "team Elo rating",
    "opponent_elo": "opponent Elo rating",
    "elo_diff": "Elo rating advantage",
    "form_5": "recent form (last 5)",
    "form_10": "recent form (last 10)",
    "form_diff_5": "form advantage (last 5)",
    "form_diff_10": "form advantage (last 10)",
    "goals_for_avg_5": "goals scored (last 5)",
    "goals_against_avg_5": "goals conceded (last 5)",
    "goals_for_avg_10": "goals scored (last 10)",
    "goals_against_avg_10": "goals conceded (last 10)",
    "shots_avg_5": "shots per game (last 5)",
    "shots_on_target_avg_5": "shots on target (last 5)",
    "xg_avg_5": "expected goals (last 5)",
    "xg_diff_5": "xG advantage (last 5)",
    "corners_avg_5": "corners (last 5)",
    "fouls_avg_5": "fouls (last 5)",
    "possession_pct_avg_5": "possession % (last 5)",
    "tournament_form_5": "tournament form (last 5)",
    "tournament_goals_for_avg_5": "tournament goals scored",
    "tournament_goals_against_avg_5": "tournament goals conceded",
    "team_rest_days": "days since last match",
    "rest_days_diff": "rest advantage",
    "implied_prob_win": "bookmaker win probability",
    "implied_prob_draw": "bookmaker draw probability",
    "implied_prob_loss": "bookmaker loss probability",
    "implied_prob_diff": "bookmaker probability edge",
    "fbref_goals_per90": "goals per 90 (FBref)",
    "fbref_xg_per90": "xG per 90 (FBref)",
    "fbref_assists_per90": "assists per 90 (FBref)",
    "h2h_matches": "head-to-head history",
    "h2h_wins": "head-to-head wins",
    "h2h_draws": "head-to-head draws",
    "h2h_losses": "head-to-head losses",
    "h2h_goals_for": "H2H goals scored",
    "h2h_goals_against": "H2H goals conceded",
}


class MatchPredictor:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or MODELS_DIR
        self.model = joblib.load(self.models_dir / "match_outcome.pkl")
        self.calibrator = joblib.load(self.models_dir / "calibrator.pkl")
        with open(self.models_dir / "feature_columns.json") as f:
            self.feature_columns: list[str] = json.load(f)
        with open(self.models_dir / "team_encoder.json") as f:
            self.teams: list[str] = json.load(f)
        self._features_df = self._load_features()
        self._team_snapshots = get_latest_team_features(self._features_df)
        self._explainer = None

    def _load_features(self) -> pd.DataFrame:
        if DATA_PATH.exists():
            return pd.read_parquet(DATA_PATH)
        return build_features()

    @property
    def explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            xgb_model = self.model.named_steps["model"]
            self._explainer = shap.TreeExplainer(xgb_model)
        return self._explainer

    def _build_feature_row(
        self,
        team_a: str,
        team_b: str,
        stage: str = "group",
        is_home: int = 1,
    ) -> pd.DataFrame:
        defaults = {c: 0.0 for c in self.feature_columns}
        snap_a = self._team_snapshots.get(team_a)
        snap_b = self._team_snapshots.get(team_b)

        if snap_a is not None:
            for col in self.feature_columns:
                if col in snap_a.index and pd.notna(snap_a[col]):
                    defaults[col] = snap_a[col]

        if snap_b is not None:
            defaults["opponent_elo"] = snap_b.get("team_elo", snap_b.get("opponent_elo", INITIAL_ELO))
            if "team_elo" in defaults:
                defaults["elo_diff"] = defaults.get("team_elo", INITIAL_ELO) - defaults["opponent_elo"]
            for diff_col, base_col in [("form_diff_5", "form_5"), ("form_diff_10", "form_10"), ("xg_diff_5", "xg_avg_5")]:
                if base_col in self.feature_columns:
                    defaults[diff_col] = defaults.get(base_col, 0) - snap_b.get(base_col, 0)
            defaults["opp_rest_days"] = snap_b.get("team_rest_days", 14)
            defaults["rest_days_diff"] = defaults.get("team_rest_days", 14) - defaults["opp_rest_days"]

        defaults["is_home"] = is_home
        defaults["is_neutral"] = int(is_home == 0)
        defaults["is_knockout"] = int(stage not in ("group",))
        stage_map = {"group": 0, "round_of_16": 1, "quarter_final": 2, "semi_final": 3, "final": 4}
        defaults["stage_encoded"] = stage_map.get(stage, 0)

        if defaults.get("team_elo", 0) == 0:
            defaults["team_elo"] = INITIAL_ELO
        if defaults.get("opponent_elo", 0) == 0:
            defaults["opponent_elo"] = INITIAL_ELO
        defaults["elo_diff"] = defaults["team_elo"] - defaults["opponent_elo"]

        h2h = self._features_df[
            ((self._features_df["team"] == team_a) & (self._features_df["opponent"] == team_b))
            | ((self._features_df["team"] == team_b) & (self._features_df["opponent"] == team_a))
        ].tail(10)
        if not h2h.empty:
            wins = draws = losses = gf = ga = 0
            for _, p in h2h.iterrows():
                if p["team"] == team_a:
                    gf += p["goals_for"]
                    ga += p["goals_against"]
                    wins += int(p["outcome"] == 2)
                    draws += int(p["outcome"] == 1)
                    losses += int(p["outcome"] == 0)
                else:
                    gf += p["goals_against"]
                    ga += p["goals_for"]
                    wins += int(p["outcome"] == 0)
                    draws += int(p["outcome"] == 1)
                    losses += int(p["outcome"] == 2)
            defaults.update({
                "h2h_matches": len(h2h),
                "h2h_wins": wins,
                "h2h_draws": draws,
                "h2h_losses": losses,
                "h2h_goals_for": gf,
                "h2h_goals_against": ga,
            })

        row = {c: defaults.get(c, 0.0) for c in self.feature_columns}
        return pd.DataFrame([row])

    def predict(
        self,
        team_a: str,
        team_b: str,
        stage: str = "group",
        is_home: int = 1,
    ) -> dict:
        X = self._build_feature_row(team_a, team_b, stage, is_home)
        proba = self.calibrator.predict_proba(X)[0]
        pred_class = int(proba.argmax())
        labels = ["Loss", "Draw", "Win"]
        shap_data = self.explain_prediction(X, pred_class)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "probabilities": {
                "loss": float(proba[0]),
                "draw": float(proba[1]),
                "win": float(proba[2]),
            },
            "predicted_outcome": labels[pred_class],
            "confidence": float(proba[pred_class]),
            "shap": shap_data,
            "narrative": generate_shap_narrative(
                shap_data, team_a, team_b, labels[pred_class], float(proba[pred_class])
            ),
            "sparse_data_warning": team_a not in self._team_snapshots or team_b not in self._team_snapshots,
        }

    def explain_prediction(self, X: pd.DataFrame, predicted_class: int) -> list[dict]:
        scaled = self.model.named_steps["imputer"].transform(X)
        scaled = self.model.named_steps["scaler"].transform(scaled)
        shap_values = self.explainer.shap_values(scaled)
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class][0]
        else:
            class_shap = shap_values[0, :, predicted_class]

        feature_importance = sorted(
            zip(self.feature_columns, class_shap),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]

        return [
            {
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "shap_value": float(val),
                "direction": "favors" if val > 0 else "against",
            }
            for feat, val in feature_importance
        ]


def generate_shap_narrative(
    shap_data: list[dict],
    team_a: str,
    team_b: str,
    outcome: str,
    confidence: float,
) -> str:
    if outcome == "Win":
        intro = f"The model favors **{team_a}** to win ({confidence:.0%}) primarily because:"
    elif outcome == "Loss":
        intro = f"The model favors **{team_b}** ({confidence:.0%} chance {team_a} loses) primarily because:"
    else:
        intro = f"The model predicts a **draw** ({confidence:.0%}) primarily because:"

    reasons = []
    for item in shap_data[:3]:
        label = item["label"]
        val = item["shap_value"]
        if abs(val) < 0.01:
            continue
        if val > 0:
            reasons.append(f"{label} favors {team_a}")
        else:
            reasons.append(f"{label} favors {team_b}")

    if not reasons:
        return intro + " historical patterns in the training data."
    return intro + " " + ", ".join(reasons) + "."


@lru_cache(maxsize=1)
def get_predictor() -> MatchPredictor:
    return MatchPredictor()
