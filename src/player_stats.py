"""Player-level team aggregates from FBref (soccerdata) with goalscorers fallback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import INTERNATIONAL_DIR
from src.team_mapping import normalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FBREF_CACHE = PROJECT_ROOT / "data" / "raw" / "fbref"
GOALSCORERS_PATH = INTERNATIONAL_DIR / "goalscorers.csv"

FBREF_SEASONS = ["2018", "2022", "2024", "2021", "2016"]
CACHE_PARQUET = "team_tournament_stats.parquet"
CACHE_CSV = "team_tournament_stats.csv"


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _read_cache(cache_dir: Path) -> pd.DataFrame | None:
    parquet_path = cache_dir / CACHE_PARQUET
    csv_path = cache_dir / CACHE_CSV
    if parquet_path.exists() and _parquet_available():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def _write_cache(df: pd.DataFrame, cache_dir: Path) -> None:
    if df.empty:
        return
    if _parquet_available():
        try:
            df.to_parquet(cache_dir / CACHE_PARQUET, index=False)
            return
        except Exception:
            pass
    df.to_csv(cache_dir / CACHE_CSV, index=False)


def _load_goalscorers_fallback() -> pd.DataFrame:
    """Team goals-per-match rates from martj42 goalscorers.csv."""
    if not GOALSCORERS_PATH.exists():
        return pd.DataFrame()

    gs = pd.read_csv(GOALSCORERS_PATH, parse_dates=["date"], low_memory=False)
    gs["team"] = gs["team"].apply(lambda x: normalize_team_name(str(x), "international"))
    gs["year"] = gs["date"].dt.year

    agg = (
        gs.groupby(["team", "year"])
        .agg(goals_scored=("team", "count"))
        .reset_index()
    )
    # Approximate per-90 as goals per calendar year / 10 matches
    agg["fbref_goals_per90"] = agg["goals_scored"] / 10.0
    agg["fbref_xg_per90"] = agg["fbref_goals_per90"] * 0.85
    agg["fbref_assists_per90"] = agg["fbref_goals_per90"] * 0.4
    return agg.rename(columns={"year": "tournament_year"})


def fetch_fbref_team_stats(force: bool = False, use_fbref: bool = False) -> pd.DataFrame:
    """Fetch or load cached FBref international squad aggregates."""
    FBREF_CACHE.mkdir(parents=True, exist_ok=True)

    if not force:
        cached = _read_cache(FBREF_CACHE)
        if cached is not None and not cached.empty:
            return cached

    if not use_fbref:
        df = _load_goalscorers_fallback()
        _write_cache(df, FBREF_CACHE)
        return df

    records: list[dict] = []
    try:
        import soccerdata as sd

        for season in FBREF_SEASONS:
            season_cache = FBREF_CACHE / f"stats_{season}.json"
            if season_cache.exists() and not force:
                with open(season_cache, encoding="utf-8") as f:
                    records.extend(json.load(f))
                continue

            season_records: list[dict] = []
            for league in ("INT-World Cup", "INT-European Championship"):
                try:
                    fbref = sd.FBref(leagues=league, seasons=season)
                    schedule = fbref.read_schedule()
                    if schedule is None or schedule.empty:
                        continue
                    team_stats = fbref.read_team_match_stats(stat_type="schedule")
                    if team_stats is None or team_stats.empty:
                        continue
                    # Aggregate by team if multi-index
                    ts = team_stats.reset_index() if isinstance(team_stats.index, pd.MultiIndex) else team_stats.copy()
                    team_col = next((c for c in ts.columns if "team" in str(c).lower()), None)
                    if team_col is None:
                        continue
                    for team_name, grp in ts.groupby(team_col):
                        team = normalize_team_name(str(team_name), "international")
                        goals = _safe_mean(grp, ["Gls", "Goals", "goals"])
                        xg = _safe_mean(grp, ["xG", "xg"])
                        ast = _safe_mean(grp, ["Ast", "Assists", "assists"])
                        season_records.append({
                            "team": team,
                            "tournament_year": int(season),
                            "fbref_goals_per90": goals or 0.0,
                            "fbref_xg_per90": xg or goals or 0.0,
                            "fbref_assists_per90": ast or 0.0,
                        })
                except Exception:
                    continue

            if season_records:
                season_cache.write_text(json.dumps(season_records), encoding="utf-8")
                records.extend(season_records)
    except ImportError:
        pass

    if not records:
        df = _load_goalscorers_fallback()
        _write_cache(df, FBREF_CACHE)
        return df

    df = pd.DataFrame(records).drop_duplicates(subset=["team", "tournament_year"])
    _write_cache(df, FBREF_CACHE)
    return df


def _safe_mean(df: pd.DataFrame, col_candidates: list[str]) -> float | None:
    for col in col_candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                return float(vals.mean())
    return None


def join_player_stats_to_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Join FBref (or fallback) stats to perspective rows by team + year."""
    stats = fetch_fbref_team_stats()
    result = df.copy()
    result["tournament_year"] = pd.to_datetime(result["match_date"]).dt.year

    if stats.empty:
        result["fbref_goals_per90"] = np.nan
        result["fbref_xg_per90"] = np.nan
        result["fbref_assists_per90"] = np.nan
        return result

    merged = result.merge(
        stats[["team", "tournament_year", "fbref_goals_per90", "fbref_xg_per90", "fbref_assists_per90"]],
        on=["team", "tournament_year"],
        how="left",
    )

    # Forward-fill from most recent year per team
    for col in ["fbref_goals_per90", "fbref_xg_per90", "fbref_assists_per90"]:
        if col in merged.columns:
            merged[col] = merged.groupby("team")[col].transform(
                lambda s: s.ffill().bfill()
            )

    return merged
