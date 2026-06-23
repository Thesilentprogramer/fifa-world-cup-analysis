#!/usr/bin/env python3
"""Train shot-level xG model from StatsBomb events."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env_check import ensure_project_deps

ensure_project_deps()

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.data_loader import load_statsbomb_shots
from src.xg_engine import prepare_shot_features, save_xg_model, train_shot_xg_model


def main() -> None:
    shots = load_statsbomb_shots()
    print(f"Loaded {len(shots)} shots from StatsBomb events")
    if shots.empty:
        print("No shots found. Run: python scripts/download_data.py --events-only")
        sys.exit(1)

    model = train_shot_xg_model(shots)
    X, y = prepare_shot_features(shots)
    proba = model.predict_proba(X)[:, 1]

    metrics = {
        "log_loss": float(log_loss(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "n_shots": len(shots),
        "goal_rate": float(y.mean()),
        "mean_predicted_xg": float(proba.mean()),
        "mean_statsbomb_xg": float(shots["statsbomb_xg"].mean()),
    }
    print(f"  log_loss={metrics['log_loss']:.4f}, brier={metrics['brier']:.4f}, auc={metrics['roc_auc']:.4f}")
    print(f"  mean predicted xG={metrics['mean_predicted_xg']:.3f} vs StatsBomb={metrics['mean_statsbomb_xg']:.3f}")

    save_xg_model(model, metrics)
    print(f"Saved -> models/xg_model.pkl")


if __name__ == "__main__":
    main()
