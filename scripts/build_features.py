#!/usr/bin/env python3
"""Build processed feature table from raw data."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env_check import ensure_project_deps

ensure_project_deps()

from src.feature_engineering import build_features


def main() -> None:
    df = build_features()
    print(f"Built {len(df)} feature rows -> data/processed/match_features.parquet")
    print(f"Teams: {df['team'].nunique()}, Date range: {df['match_date'].min()} to {df['match_date'].max()}")
    print(f"Outcome distribution:\n{df['outcome'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
