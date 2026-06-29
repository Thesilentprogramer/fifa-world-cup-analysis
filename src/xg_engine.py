"""Shot-level xG model and match simulation (Phase 2)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
XG_MODEL_PATH = MODELS_DIR / "xg_model.pkl"
XG_META_PATH = MODELS_DIR / "xg_model_meta.json"

SHOT_FEATURE_COLS = ["distance", "angle", "body_part", "technique", "shot_type"]
CATEGORICAL = ["body_part", "technique", "shot_type"]
NUMERIC = ["distance", "angle"]


def _load_shots(shots: pd.DataFrame | None) -> pd.DataFrame:
    if shots is not None:
        return shots
    from src.statsbomb_shots import load_statsbomb_shots
    return load_statsbomb_shots()


def prepare_shot_features(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X, y from raw shot dataframe."""
    df = shots.dropna(subset=["distance", "angle", "is_goal"]).copy()
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").astype(str)
    X = df[SHOT_FEATURE_COLS]
    y = df["is_goal"].astype(int)
    return X, y


def train_shot_xg_model(shots: pd.DataFrame | None = None) -> Pipeline:
    """Train logistic pipeline on StatsBomb shots."""
    shots = _load_shots(shots)
    if shots.empty:
        raise ValueError("No shot data found. Run: python scripts/download_data.py --events-only")

    X, y = prepare_shot_features(shots)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    from sklearn.linear_model import LogisticRegression

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    model.fit(X, y)
    return model


def save_xg_model(model: Pipeline, metrics: dict | None = None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, XG_MODEL_PATH)
    meta = {"feature_columns": SHOT_FEATURE_COLS, "metrics": metrics or {}}
    XG_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def load_xg_model() -> Pipeline:
    from src.env_check import ensure_model_compatibility
    ensure_model_compatibility()
    if not XG_MODEL_PATH.exists():
        raise FileNotFoundError(
            "xG model not found. Run: python scripts/train_xg_model.py"
        )
    return joblib.load(XG_MODEL_PATH)


def predict_shot_xg(shots_df: pd.DataFrame, model: Pipeline | None = None) -> np.ndarray:
    """Predict goal probability per shot."""
    model = model or load_xg_model()
    df = shots_df.copy()
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    for col in NUMERIC:
        if col not in df.columns:
            df[col] = 0.0
    return model.predict_proba(df[SHOT_FEATURE_COLS])[:, 1]


def team_xg_rates_from_history(
    team: str,
    shots: pd.DataFrame | None = None,
    model: Pipeline | None = None,
) -> dict[str, float]:
    """Estimate per-team attacking xG rate and defensive concession from shot history."""
    shots = _load_shots(shots)
    model = model or load_xg_model()

    team_shots = shots[shots["team"] == team]
    against = shots[shots["match_id"].isin(team_shots["match_id"]) & (shots["team"] != team)]

    if team_shots.empty:
        return {"xg_for_rate": 1.2, "xg_against_rate": 1.2, "n_matches": 0}

    xg_for = predict_shot_xg(team_shots, model).sum()
    xg_against = predict_shot_xg(against, model).sum() if not against.empty else xg_for
    n_matches = team_shots["match_id"].nunique() or 1

    return {
        "xg_for_rate": float(xg_for / n_matches),
        "xg_against_rate": float(xg_against / n_matches),
        "n_matches": int(n_matches),
    }


def simulate_match(
    team_a: str,
    team_b: str,
    n_simulations: int = 10_000,
    shots: pd.DataFrame | None = None,
    model: Pipeline | None = None,
) -> dict:
    """Monte Carlo scoreline simulation from historical xG rates."""
    shots = _load_shots(shots)
    model = model or load_xg_model()

    ra = team_xg_rates_from_history(team_a, shots, model)
    rb = team_xg_rates_from_history(team_b, shots, model)

    # Poisson goals from blended xG rates
    lam_a = max(0.3, (ra["xg_for_rate"] + rb["xg_against_rate"]) / 2)
    lam_b = max(0.3, (rb["xg_for_rate"] + ra["xg_against_rate"]) / 2)

    goals_a = np.random.poisson(lam_a, n_simulations)
    goals_b = np.random.poisson(lam_b, n_simulations)

    scorelines: dict[tuple[int, int], int] = {}
    for ga, gb in zip(goals_a, goals_b):
        key = (int(ga), int(gb))
        scorelines[key] = scorelines.get(key, 0) + 1

    top = sorted(scorelines.items(), key=lambda x: x[1], reverse=True)[:10]
    win_a = float((goals_a > goals_b).mean())
    draw = float((goals_a == goals_b).mean())
    win_b = float((goals_a < goals_b).mean())

    return {
        "team_a": team_a,
        "team_b": team_b,
        "expected_xg_a": lam_a,
        "expected_xg_b": lam_b,
        "prob_win_a": win_a,
        "prob_draw": draw,
        "prob_win_b": win_b,
        "top_scorelines": [
            {"score_a": k[0], "score_b": k[1], "probability": v / n_simulations}
            for k, v in top
        ],
        "n_simulations": n_simulations,
    }
