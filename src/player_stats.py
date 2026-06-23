"""Player-level team aggregates from FBref (soccerdata) with goalscorers fallback.

Phase 3 additions:
- load_top_scorers()         — all-time/filtered international top scorers from martj42 goalscorers.csv
- aggregate_statsbomb_player_stats() — event-level player stats from cached StatsBomb JSON files
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import INTERNATIONAL_DIR
from src.team_mapping import normalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FBREF_CACHE = PROJECT_ROOT / "data" / "raw" / "fbref"
STATSBOMB_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb"
GOALSCORERS_PATH = INTERNATIONAL_DIR / "goalscorers.csv"

FBREF_SEASONS = ["2018", "2022", "2024", "2021", "2016"]
CACHE_PARQUET = "team_tournament_stats.parquet"
CACHE_CSV = "team_tournament_stats.csv"

# --- Top Scorers ---------------------------------------------------------------

def load_top_scorers(
    min_year: int | None = None,
    max_year: int | None = None,
    team: str | None = None,
    include_penalties: bool = True,
    include_own_goals: bool = False,
    top_n: int = 50,
) -> pd.DataFrame:
    """Load all-time international top scorers from martj42 goalscorers.csv.

    Returns a DataFrame with columns: scorer, team, goals, penalties, matches.
    """
    if not GOALSCORERS_PATH.exists():
        return pd.DataFrame(columns=["scorer", "team", "goals", "penalty_goals", "matches"])

    gs = pd.read_csv(GOALSCORERS_PATH, parse_dates=["date"], low_memory=False)
    gs["team"] = gs["team"].apply(lambda x: normalize_team_name(str(x), "international"))
    gs["year"] = gs["date"].dt.year

    # Filters
    if min_year:
        gs = gs[gs["year"] >= min_year]
    if max_year:
        gs = gs[gs["year"] <= max_year]
    if team:
        norm_team = normalize_team_name(str(team), "international")
        gs = gs[gs["team"] == norm_team]

    # Exclude own goals from goal count unless requested
    if not include_own_goals:
        gs_goals = gs[gs["own_goal"].astype(str).str.lower().isin(["false", "0", ""])]
    else:
        gs_goals = gs.copy()

    # Separate penalty from non-penalty
    is_penalty = gs_goals["penalty"].astype(str).str.lower().isin(["true", "1"])
    penalty_goals = gs_goals[is_penalty]
    non_penalty_goals = gs_goals[~is_penalty]

    if include_penalties:
        agg_df = gs_goals
    else:
        agg_df = non_penalty_goals

    counts = (
        agg_df.groupby(["scorer", "team"])
        .agg(goals=("scorer", "count"))
        .reset_index()
    )
    pen_counts = (
        penalty_goals.groupby(["scorer", "team"])
        .agg(penalty_goals=("scorer", "count"))
        .reset_index()
    )
    # Unique match appearances
    match_counts = (
        gs_goals.groupby(["scorer", "team"])
        .agg(matches=("date", "nunique"))
        .reset_index()
    )

    result = counts.merge(pen_counts, on=["scorer", "team"], how="left")
    result = result.merge(match_counts, on=["scorer", "team"], how="left")
    result["penalty_goals"] = result["penalty_goals"].fillna(0).astype(int)
    result["matches"] = result["matches"].fillna(0).astype(int)
    result = result.sort_values("goals", ascending=False).head(top_n).reset_index(drop=True)
    return result


# --- StatsBomb Player Event Aggregates -----------------------------------------

def _nested_name(val) -> str:
    if isinstance(val, dict):
        return val.get("name", "") or ""
    if isinstance(val, str):
        return val
    return ""


def _parse_events_file(path: Path) -> list:
    try:
        text = path.read_text(encoding="utf-8").replace("NaN", "null")
        return json.loads(text)
    except Exception:
        return []


def aggregate_statsbomb_player_stats(data_dir: Path | None = None) -> pd.DataFrame:
    """Aggregate player-level event stats from cached StatsBomb JSON files.

    Columns: player, team, competition, season, goals, shots, xG, key_passes,
             assists, passes, completed_passes, dribbles, completed_dribbles,
             tackles, interceptions, xg_chain (xG of possessions player was involved in).
    """
    base = data_dir or STATSBOMB_DIR
    if not base.exists():
        return pd.DataFrame()

    # (player, team, comp, season) -> stats dict
    records: dict[tuple, dict] = {}
    # possession_id -> xg: for xG chain
    possession_xg: dict[tuple, float] = {}  # (comp, season, match_id, poss_id) -> cumulative xg

    def _key(player: str, team: str, comp: str, season: str):
        return (player, team, comp, season)

    def _get(k):
        if k not in records:
            records[k] = {
                "player": k[0], "team": k[1], "competition": k[2], "season": k[3],
                "goals": 0, "shots": 0, "xG": 0.0,
                "key_passes": 0, "assists": 0,
                "passes": 0, "completed_passes": 0,
                "dribbles": 0, "completed_dribbles": 0,
                "tackles": 0, "interceptions": 0,
                "xg_chain_possessions": set(),  # possession ids involved in
                "xg_chain": 0.0,
            }
        return records[k]

    for comp_dir in sorted(base.iterdir()):
        if not comp_dir.is_dir():
            continue
        for season_dir in sorted(comp_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            events_dir = season_dir / "events_by_match"
            matches_file = season_dir / "matches.json"
            if not events_dir.exists():
                continue

            # Load competition / season metadata from matches.json
            comp_name = comp_dir.name
            season_name = season_dir.name
            if matches_file.exists():
                try:
                    ms = json.loads(matches_file.read_text(encoding="utf-8"))
                    if ms:
                        m0 = ms[0]
                        c = m0.get("competition", {})
                        s = m0.get("season", {})
                        comp_name = (c.get("competition_name", "") if isinstance(c, dict) else str(c)) or comp_dir.name
                        season_name = (s.get("season_name", "") if isinstance(s, dict) else str(s)) or season_dir.name
                except Exception:
                    pass

            # First pass: build possession -> shot xg map
            match_poss_xg: dict[str, dict[int, float]] = {}
            for ef in sorted(events_dir.glob("*.json")):
                evts = _parse_events_file(ef)
                mid = ef.stem
                pxg: dict[int, float] = {}
                for e in evts:
                    etype = _nested_name(e.get("type", ""))
                    poss_id = e.get("possession")
                    if etype == "Shot" and poss_id is not None:
                        shot = e.get("shot")
                        xg_val = 0.0
                        if isinstance(shot, dict):
                            xg_val = float(shot.get("statsbomb_xg", 0) or 0)
                        else:
                            raw = e.get("shot_statsbomb_xg")
                            if raw is not None and str(raw) != "nan":
                                xg_val = float(raw)
                        pxg[poss_id] = pxg.get(poss_id, 0.0) + xg_val
                match_poss_xg[mid] = pxg

            # Second pass: aggregate per-player stats
            for ef in sorted(events_dir.glob("*.json")):
                evts = _parse_events_file(ef)
                mid = ef.stem
                pxg = match_poss_xg.get(mid, {})

                for e in evts:
                    player_raw = e.get("player")
                    player = _nested_name(player_raw) if player_raw else ""
                    if not player:
                        continue
                    team_raw = e.get("team")
                    team = normalize_team_name(_nested_name(team_raw)) if team_raw else ""
                    if not team:
                        continue

                    etype = _nested_name(e.get("type", ""))
                    k = _key(player, team, comp_name, season_name)
                    st = _get(k)

                    poss_id = e.get("possession")

                    # Track xG chain: any touch in a possession with a shot
                    if poss_id is not None and poss_id in pxg:
                        if poss_id not in st["xg_chain_possessions"]:
                            st["xg_chain_possessions"].add(poss_id)
                            st["xg_chain"] += pxg[poss_id]

                    if etype == "Shot":
                        st["shots"] += 1
                        shot = e.get("shot")
                        outcome = ""
                        xg_val = 0.0
                        if isinstance(shot, dict):
                            outcome = _nested_name(shot.get("outcome", ""))
                            xg_val = float(shot.get("statsbomb_xg", 0) or 0)
                        else:
                            raw_out = e.get("shot_outcome")
                            if raw_out:
                                outcome = str(raw_out)
                            raw_xg = e.get("shot_statsbomb_xg")
                            if raw_xg is not None and str(raw_xg) != "nan":
                                xg_val = float(raw_xg)
                        if outcome == "Goal":
                            st["goals"] += 1
                        st["xG"] += xg_val

                    elif etype == "Pass":
                        st["passes"] += 1
                        pass_data = e.get("pass", {})
                        if isinstance(pass_data, dict):
                            # Completed: no outcome (miscontrol/blocked sets outcome)
                            if "outcome" not in pass_data:
                                st["completed_passes"] += 1
                            # Key pass: pass that directly led to a shot
                            shot_assist = pass_data.get("shot_assist")
                            goal_assist = pass_data.get("goal_assist")
                            if shot_assist or goal_assist:
                                st["key_passes"] += 1
                            if goal_assist:
                                st["assists"] += 1

                    elif etype == "Dribble":
                        st["dribbles"] += 1
                        drib = e.get("dribble", {})
                        if isinstance(drib, dict):
                            if _nested_name(drib.get("outcome", "")) == "Complete":
                                st["completed_dribbles"] += 1

                    elif etype == "Tackle":
                        st["tackles"] += 1
                    elif etype == "Interception":
                        st["interceptions"] += 1

    if not records:
        return pd.DataFrame()

    rows = []
    for st in records.values():
        rows.append({
            "player": st["player"],
            "team": st["team"],
            "competition": st["competition"],
            "season": st["season"],
            "goals": st["goals"],
            "shots": st["shots"],
            "xG": round(st["xG"], 3),
            "key_passes": st["key_passes"],
            "assists": st["assists"],
            "passes": st["passes"],
            "completed_passes": st["completed_passes"],
            "dribbles": st["dribbles"],
            "completed_dribbles": st["completed_dribbles"],
            "tackles": st["tackles"],
            "interceptions": st["interceptions"],
            "xg_chain": round(st["xg_chain"], 3),
        })

    df = pd.DataFrame(rows)
    df = df[df["shots"] + df["passes"] + df["dribbles"] + df["tackles"] > 0]
    return df.sort_values("goals", ascending=False).reset_index(drop=True)


# --- FBref team-level aggregates (Phase 1 code kept) ---------------------------

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

    for col in ["fbref_goals_per90", "fbref_xg_per90", "fbref_assists_per90"]:
        if col in merged.columns:
            merged[col] = merged.groupby("team")[col].transform(
                lambda s: s.ffill().bfill()
            )

    return merged
