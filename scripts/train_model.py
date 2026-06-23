#!/usr/bin/env python3
"""Train match outcome model with calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env_check import ensure_project_deps

ensure_project_deps()

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.feature_engineering import FEATURE_COLUMNS

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "match_features.parquet"

VAL_START = pd.Timestamp("2018-06-01")
VAL_END = pd.Timestamp("2018-07-31")
TEST_START = pd.Timestamp("2022-11-01")
TEST_END = pd.Timestamp("2022-12-31")


def load_splits():
    df = pd.read_parquet(DATA_PATH)
    df["match_date"] = pd.to_datetime(df["match_date"])
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    # Drop columns with no usable values (e.g. odds when download failed)
    feature_cols = [
        c for c in feature_cols
        if df[c].notna().any() and df[c].nunique(dropna=True) > 1
    ]

    is_mens_wc = df["competition"].str.contains("FIFA World Cup", case=False, na=False)
    is_wc_2018 = (df["match_date"] >= VAL_START) & (df["match_date"] <= VAL_END) & is_mens_wc
    is_wc_2022 = (df["match_date"] >= TEST_START) & (df["match_date"] <= TEST_END) & is_mens_wc

    train = df[~is_wc_2018 & ~is_wc_2022]
    val = df[is_wc_2018]
    test = df[is_wc_2022]
    return train, val, test, feature_cols


def evaluate(name: str, y_true: np.ndarray, proba: np.ndarray) -> dict:
    ll = log_loss(y_true, proba, labels=[0, 1, 2])
    acc = accuracy_score(y_true, proba.argmax(axis=1))
    brier = np.mean([
        brier_score_loss((y_true == c).astype(int), proba[:, c], pos_label=1)
        for c in range(3)
    ])
    print(f"  {name}: log_loss={ll:.4f}, brier={brier:.4f}, accuracy={acc:.4f}")
    return {"log_loss": ll, "brier": brier, "accuracy": acc}


def high_confidence_accuracy(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.55) -> float:
    max_p = proba.max(axis=1)
    mask = max_p >= threshold
    if mask.sum() == 0:
        return 0.0
    return float(accuracy_score(y_true[mask], proba.argmax(axis=1)[mask]))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test, feature_cols = load_splits()

    X_train, y_train = train[feature_cols], train["outcome"]
    X_val, y_val = val[feature_cols], val["outcome"]
    X_test, y_test = test[feature_cols], test["outcome"]

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Features: {len(feature_cols)}")

    sample_weights = compute_sample_weight("balanced", y_train)

    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ])
    lr_pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
    print("\nBaseline Logistic Regression:")
    evaluate("val", y_val.values, lr_pipe.predict_proba(X_val))

    xgb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            max_depth=5,
            learning_rate=0.08,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )),
    ])
    xgb_pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
    best_pipe = xgb_pipe

    print("\nXGBoost (uncalibrated):")
    val_proba_raw = best_pipe.predict_proba(X_val)
    evaluate("val", y_val.values, val_proba_raw)

    calibrated = CalibratedClassifierCV(FrozenEstimator(best_pipe), method="sigmoid")
    calibrated.fit(X_val, y_val)

    print("\nXGBoost (calibrated on 2018 WC):")
    val_proba_cal = calibrated.predict_proba(X_val)
    test_proba_cal = calibrated.predict_proba(X_test)
    val_metrics = evaluate("val", y_val.values, val_proba_cal)

    print("\nHeld-out 2022 WC test:")
    test_metrics = evaluate("test", y_test.values, test_proba_cal)
    hc_acc = high_confidence_accuracy(y_test.values, test_proba_cal)
    print(f"  high_confidence_accuracy (>=55%): {hc_acc:.4f} on {(test_proba_cal.max(axis=1) >= 0.55).sum()} matches")

    joblib.dump(best_pipe, MODELS_DIR / "match_outcome.pkl")
    joblib.dump(calibrated, MODELS_DIR / "calibrator.pkl")

    teams = sorted(set(train["team"]) | set(train["opponent"]))
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS_DIR / "team_encoder.json", "w") as f:
        json.dump(teams, f, indent=2)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics, "high_confidence_test_acc": hc_acc}, f, indent=2)

    print(f"\nSaved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
