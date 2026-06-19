"""Phase 5: Model vs betting odds comparison utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def odds_to_implied_probs(home_odds: float, draw_odds: float, away_odds: float) -> np.ndarray:
    """Convert decimal odds to normalized implied probabilities (vig removed)."""
    raw = np.array([1 / home_odds, 1 / draw_odds, 1 / away_odds])
    return raw / raw.sum()


def benchmark_vs_odds(
    model_probs: np.ndarray,
    odds_probs: np.ndarray,
    actual_outcomes: np.ndarray,
) -> dict[str, float]:
    """Return log-loss, Brier, accuracy for model vs bookmaker on same matches."""
    model_ll = log_loss(actual_outcomes, model_probs, labels=[0, 1, 2])
    odds_ll = log_loss(actual_outcomes, odds_probs, labels=[0, 1, 2])
    model_brier = np.mean(np.sum((model_probs - np.eye(3)[actual_outcomes]) ** 2, axis=1))
    odds_brier = np.mean(np.sum((odds_probs - np.eye(3)[actual_outcomes]) ** 2, axis=1))
    model_acc = accuracy_score(actual_outcomes, model_probs.argmax(axis=1))
    odds_acc = accuracy_score(actual_outcomes, odds_probs.argmax(axis=1))

    return {
        "model_log_loss": model_ll,
        "odds_log_loss": odds_ll,
        "model_brier": model_brier,
        "odds_brier": odds_brier,
        "model_accuracy": model_acc,
        "odds_accuracy": odds_acc,
    }
