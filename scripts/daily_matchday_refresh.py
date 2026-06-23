#!/usr/bin/env python3
"""Daily matchday refresh: sync results, update prediction log, conditional retrain."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env_check import ensure_project_deps

ensure_project_deps()

import pandas as pd

from scripts.download_data import download_international
from src.api_football_client import load_wc_fixtures, sync_wc_fixtures
from src.feature_engineering import build_features
from src.prediction_cache import load_prediction_log, update_played_results, ensure_played_predictions
from src.prematch_analysis import load_upcoming_today_tomorrow

LAST_REFRESH_PATH = PROJECT_ROOT / "data" / "processed" / "last_refresh.json"


def _load_last_refresh() -> dict:
    if not LAST_REFRESH_PATH.exists():
        return {}
    return json.loads(LAST_REFRESH_PATH.read_text(encoding="utf-8"))


def _save_last_refresh(meta: dict) -> None:
    LAST_REFRESH_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_REFRESH_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _count_finished(fixtures: pd.DataFrame) -> int:
    if fixtures.empty:
        return 0
    return int((fixtures["is_finished"] == True).sum())  # noqa: E712


def _retrain_model() -> None:
    """Rebuild features and retrain match outcome model."""
    from scripts.train_model import main as train_main

    print("  Rebuilding features...")
    build_features()
    print("  Retraining match model...")
    train_main()


def main() -> None:
    print(f"=== Daily matchday refresh @ {datetime.now(timezone.utc).isoformat()} ===")

    prev = _load_last_refresh()
    prev_finished = int(prev.get("finished_count", 0))

    print("1. Downloading martj42 international results...")
    download_international(force=True)

    print("2. Syncing WC 2026 fixtures...")
    fixtures = sync_wc_fixtures(force=True)
    finished_count = _count_finished(fixtures)
    new_finished = finished_count - prev_finished
    print(f"   Finished matches: {finished_count} (new since last run: {new_finished})")

    print("3. Caching kickoff predictions for today/tomorrow...")
    try:
        load_upcoming_today_tomorrow(n_xg_sims=300)
    except Exception as e:
        print(f"   Warning: upcoming analysis skipped: {e}")

    print("4. Updating played prediction log...")
    added = ensure_played_predictions(fixtures)
    updated = update_played_results(fixtures)
    print(f"   Backfilled {added} predictions, updated {updated} result rows")

    retrained = False
    if new_finished > 0:
        print("5. New finished matches detected — retraining model...")
        try:
            _retrain_model()
            retrained = True
        except Exception as e:
            print(f"   ERROR during retrain: {e}")
    else:
        print("5. No new finished matches — skipping retrain.")

    meta = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "finished_count": finished_count,
        "new_finished": new_finished,
        "retrained": retrained,
        "prediction_log_rows": len(load_prediction_log()),
        "fixture_count": len(fixtures),
    }
    _save_last_refresh(meta)
    print(f"Done. Metadata -> {LAST_REFRESH_PATH}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
