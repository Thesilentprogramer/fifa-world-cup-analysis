"""Data loading for StatsBomb and Transfermarkt datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.team_mapping import normalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATSBOMB_DIR = RAW_DIR / "statsbomb"
TRANSFERMARKT_DIR = RAW_DIR / "transfermarkt"
INTERNATIONAL_DIR = RAW_DIR / "international"
ODDS_DIR = RAW_DIR / "odds"

MARTJ42_URLS = {
    "results": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "goalscorers": "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
}

INCLUDE_TOURNAMENT_KEYWORDS = [
    "world cup", "euro", "european championship", "friendly", "qualification",
    "qualifier", "nations league", "copa america", "copa américa", "africa cup",
    "asian cup", "concacaf", "gold cup", "uefa",
]

EXCLUDE_TOURNAMENT_KEYWORDS = [
    "women", "woman", "u21", "u23", "u20", "u19", "u18", "u17",
    "b team", "olympic", "youth", "junior",
]

MIN_MATCH_DATE = pd.Timestamp("2000-01-01")

# International competition IDs from StatsBomb open data
INTERNATIONAL_COMPETITIONS = {
    43: "FIFA World Cup",
    55: "UEFA Euro",
    16: "Champions League",  # included for richer team stats if needed
}

# Legacy CSV paths (repo structure changed; download skips missing files gracefully)
TRANSFERMARKT_FILES = {
    "games": "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/prep/games.csv",
    "players": "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/prep/players.csv",
    "appearances": "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/prep/appearances.csv",
    "game_lineups": "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/prep/game_lineups.csv",
    "player_valuations": "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/prep/player_valuations.csv",
}

INTERNATIONAL_COMPETITION_KEYWORDS = [
    "world cup",
    "european championship",
    "euro",
    "copa america",
    "africa cup",
    "asian cup",
    "concacaf",
    "nations league",
    "friendly",
    "qualification",
    "qualifier",
    "international",
]


def _team_name(m: dict, side: str) -> str:
    """Extract team name from match record (flat or nested StatsBomb format)."""
    key = f"{side}_team"
    val = m.get(key, "")
    if isinstance(val, dict):
        return val.get(f"{key}_name", "") or val.get("name", "")
    if isinstance(val, str):
        return val
    return ""


def _competition_name(m: dict) -> str:
    comp = m.get("competition", "")
    if isinstance(comp, dict):
        return comp.get("competition_name", "")
    if isinstance(comp, str):
        return comp
    return m.get("competition_name", "")


def _season_name(m: dict) -> str:
    season = m.get("season", "")
    if isinstance(season, dict):
        return season.get("season_name", "")
    if isinstance(season, str):
        return season
    return ""


def _stage_name(m: dict) -> str | None:
    stage = m.get("stage", "")
    if isinstance(stage, dict):
        return stage.get("name")
    if isinstance(stage, str):
        return stage
    return None


def _nested_name(val: Any) -> str:
    """Extract name from nested dict or plain string."""
    if isinstance(val, dict):
        return val.get("name", "") or ""
    if isinstance(val, str):
        return val
    return ""


def _event_team_name(e: dict) -> str:
    return _nested_name(e.get("team", ""))


def _event_type_name(e: dict) -> str:
    return _nested_name(e.get("type", ""))


def _event_shot_outcome(e: dict) -> str:
    shot = e.get("shot")
    if isinstance(shot, dict):
        return _nested_name(shot.get("outcome", ""))
    outcome = e.get("shot_outcome")
    if outcome is not None and str(outcome) != "nan":
        return str(outcome)
    return ""


def _event_shot_xg(e: dict) -> float:
    shot = e.get("shot")
    if isinstance(shot, dict):
        return float(shot.get("statsbomb_xg", 0) or 0)
    xg = e.get("shot_statsbomb_xg")
    if xg is not None and str(xg) != "nan":
        return float(xg)
    return 0.0


def _is_corner(e: dict) -> bool:
    pas = e.get("pass")
    if isinstance(pas, dict):
        if pas.get("pass_cluster_id") == 2:
            return True
        if _nested_name(pas.get("type", "")) == "Corner":
            return True
    if e.get("pass_type") == "Corner":
        return True
    return False


def _parse_stage(stage: str | None) -> str:
    if not stage:
        return "group"
    s = stage.lower()
    if "final" in s and "semi" not in s and "quarter" not in s:
        return "final"
    if "semi" in s:
        return "semi_final"
    if "quarter" in s:
        return "quarter_final"
    if "round of 16" in s or "last 16" in s or "round of sixteen" in s:
        return "round_of_16"
    return "group"


def load_statsbomb_matches(data_dir: Path | None = None) -> pd.DataFrame:
    """Load all StatsBomb matches from cached JSON files into a unified DataFrame."""
    base = data_dir or STATSBOMB_DIR
    records: list[dict[str, Any]] = []

    if not base.exists():
        return pd.DataFrame()

    for comp_dir in sorted(base.iterdir()):
        if not comp_dir.is_dir():
            continue
        for season_dir in sorted(comp_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            matches_file = season_dir / "matches.json"
            if not matches_file.exists():
                continue
            with open(matches_file, encoding="utf-8") as f:
                matches = json.load(f)
            for m in matches:
                home = normalize_team_name(_team_name(m, "home"))
                away = normalize_team_name(_team_name(m, "away"))
                if not home or not away:
                    continue
                records.append(
                    {
                        "match_id": m.get("match_id"),
                        "competition": _competition_name(m),
                        "competition_id": m.get("competition_id"),
                        "season": _season_name(m),
                        "match_date": m.get("match_date"),
                        "home_team": home,
                        "away_team": away,
                        "home_score": m.get("home_score"),
                        "away_score": m.get("away_score"),
                        "stage": _parse_stage(_stage_name(m)),
                        "stadium": m.get("stadium", {}).get("name") if isinstance(m.get("stadium"), dict) else m.get("stadium"),
                        "source": "statsbomb",
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("match_date").reset_index(drop=True)


def load_statsbomb_match_stats(data_dir: Path | None = None) -> pd.DataFrame:
    """Aggregate per-match team stats from StatsBomb event data."""
    base = data_dir or STATSBOMB_DIR
    rows: list[dict[str, Any]] = []

    if not base.exists():
        return pd.DataFrame()

    for comp_dir in sorted(base.iterdir()):
        if not comp_dir.is_dir():
            continue
        for season_dir in sorted(comp_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            events_file = season_dir / "events.json"
            events_dir = season_dir / "events_by_match"
            matches_file = season_dir / "matches.json"
            if not matches_file.exists():
                continue

            with open(matches_file, encoding="utf-8") as f:
                matches = {m["match_id"]: m for m in json.load(f)}

            events: list = []
            if events_dir.exists() and any(events_dir.glob("*.json")):
                for match_events_file in sorted(events_dir.glob("*.json")):
                    try:
                        with open(match_events_file, encoding="utf-8") as f:
                            events.extend(json.load(f))
                    except (json.JSONDecodeError, ValueError):
                        continue
            elif events_file.exists() and events_file.stat().st_size > 0:
                try:
                    with open(events_file, encoding="utf-8") as f:
                        events = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    events = []

            if not events:
                continue

            match_events: dict[int, list] = {}
            for e in events:
                mid = e.get("match_id")
                match_events.setdefault(mid, []).append(e)

            for mid, evts in match_events.items():
                m = matches.get(mid)
                if not m:
                    continue
                home = normalize_team_name(_team_name(m, "home"))
                away = normalize_team_name(_team_name(m, "away"))

                for team, opp, is_home in [(home, away, True), (away, home, False)]:
                    team_evts = [
                        e for e in evts
                        if _event_team_name(e) and normalize_team_name(_event_team_name(e)) == team
                    ]
                    shots = [e for e in team_evts if _event_type_name(e) == "Shot"]
                    sot = [
                        e for e in shots
                        if _event_shot_outcome(e) in ("Goal", "Saved", "Saved to Post")
                    ]
                    xg = sum(_event_shot_xg(e) for e in shots)
                    corners = sum(1 for e in team_evts if _is_corner(e))
                    fouls = sum(1 for e in team_evts if _event_type_name(e) == "Foul Committed")
                    possession_evts = len(team_evts)
                    total_evts = len(evts) or 1

                    rows.append({
                        "match_id": mid,
                        "match_date": m.get("match_date"),
                        "team": team,
                        "opponent": opp,
                        "is_home": is_home,
                        "shots": len(shots),
                        "shots_on_target": len(sot),
                        "xg": xg,
                        "corners": corners,
                        "fouls": fouls,
                        "possession_pct": possession_evts / total_evts * 100,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def load_transfermarkt_games(data_dir: Path | None = None) -> pd.DataFrame:
    """Load Transfermarkt games filtered to international matches."""
    base = data_dir or TRANSFERMARKT_DIR
    games_path = base / "games.csv"
    if not games_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(games_path, low_memory=False)
    comp_col = "competition_name" if "competition_name" in df.columns else "competition"
    if comp_col not in df.columns:
        return pd.DataFrame()

    mask = df[comp_col].astype(str).str.lower().apply(
        lambda x: any(kw in x for kw in INTERNATIONAL_COMPETITION_KEYWORDS)
    )
    df = df[mask].copy()

    home_col = "home_club_name" if "home_club_name" in df.columns else "home_team_name"
    away_col = "away_club_name" if "away_club_name" in df.columns else "away_team_name"

    if home_col in df.columns:
        df["home_team"] = df[home_col].apply(lambda x: normalize_team_name(str(x), "transfermarkt"))
    if away_col in df.columns:
        df["away_team"] = df[away_col].apply(lambda x: normalize_team_name(str(x), "transfermarkt"))

    date_col = "date" if "date" in df.columns else "match_date"
    if date_col in df.columns:
        df["match_date"] = pd.to_datetime(df[date_col], errors="coerce")

    return df


def load_transfermarkt_valuations(data_dir: Path | None = None) -> pd.DataFrame:
    """Load player valuation history."""
    base = data_dir or TRANSFERMARKT_DIR
    path = base / "player_valuations.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_transfermarkt_lineups(data_dir: Path | None = None) -> pd.DataFrame:
    """Load game lineups from Transfermarkt."""
    base = data_dir or TRANSFERMARKT_DIR
    path = base / "game_lineups.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_transfermarkt_players(data_dir: Path | None = None) -> pd.DataFrame:
    """Load player metadata."""
    base = data_dir or TRANSFERMARKT_DIR
    path = base / "players.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _derive_competition(tournament: str) -> str:
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return "FIFA World Cup"
    if "qualif" in t and "world cup" in t:
        return "World Cup qualification"
    if "european championship" in t or ("euro" in t and "qualif" not in t):
        return "UEFA Euro"
    if "qualif" in t and ("euro" in t or "european" in t):
        return "Euro qualification"
    if "friendly" in t:
        return "Friendly"
    if "nations league" in t:
        return "UEFA Nations League"
    if "copa america" in t or "copa américa" in t:
        return "Copa America"
    if "africa cup" in t:
        return "Africa Cup of Nations"
    if "asian cup" in t:
        return "Asian Cup"
    if "concacaf" in t or "gold cup" in t:
        return "CONCACAF"
    return str(tournament)


def tournament_category(tournament: str) -> str:
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return "world_cup"
    if ("euro" in t or "european championship" in t) and "qualif" not in t:
        return "euro"
    if "qualif" in t:
        return "qualifier"
    if "friendly" in t:
        return "friendly"
    return "other"


def _filter_tournaments(df: pd.DataFrame) -> pd.DataFrame:
    tcol = "tournament" if "tournament" in df.columns else "competition"
    t = df[tcol].astype(str).str.lower()
    include = t.apply(lambda x: any(k in x for k in INCLUDE_TOURNAMENT_KEYWORDS))
    exclude = t.apply(lambda x: any(k in x for k in EXCLUDE_TOURNAMENT_KEYWORDS))
    return df[include & ~exclude].copy()


def load_international_results(data_dir: Path | None = None) -> pd.DataFrame:
    """Load martj42 men's international results."""
    base = data_dir or INTERNATIONAL_DIR
    path = base / "results.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    df = df.rename(columns={"date": "match_date"})
    df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(str(x), "international"))
    df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(str(x), "international"))
    df = df[df["home_team"] != df["away_team"]].copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["match_date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df[df["match_date"] >= MIN_MATCH_DATE]
    df = _filter_tournaments(df)
    df["competition"] = df["tournament"].apply(_derive_competition)
    df["tournament_category"] = df["tournament"].apply(tournament_category)
    df["is_neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    df["stage"] = "group"
    df["source"] = "international"
    df["match_id"] = (
        df["match_date"].dt.strftime("%Y%m%d")
        + "_"
        + df["home_team"].str.replace(" ", "")
        + "_"
        + df["away_team"].str.replace(" ", "")
    )
    df = df.drop_duplicates(subset=["match_id"], keep="first")
    return df.sort_values("match_date").reset_index(drop=True)


def load_odds(data_dir: Path | None = None) -> pd.DataFrame:
    """Load bookmaker odds for international tournaments."""
    base = data_dir or ODDS_DIR
    for fname in ("world_cup_football_data.csv", "world_cup_football_data.xlsx", "world_cup_odds.csv"):
        path = base / fname
        if not path.exists():
            continue
        if path.suffix == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        date_col = next((c for c in df.columns if c.lower() in ("date", "match_date")), None)
        home_col = next((c for c in df.columns if c.lower() in ("home", "hometeam", "home_team")), None)
        away_col = next((c for c in df.columns if c.lower() in ("away", "awayteam", "away_team")), None)
        if not all([date_col, home_col, away_col]):
            continue
        df = df.rename(columns={date_col: "match_date", home_col: "home_team", away_col: "away_team"})
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(str(x), "international"))
        df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(str(x), "international"))
        for col in ("B365CH", "B365CD", "B365CA", "B365H", "B365D", "B365A", "PSH", "PSD", "PSA"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        hcol = "B365CH" if "B365CH" in df.columns else ("B365H" if "B365H" in df.columns else "PSH")
        dcol = "B365CD" if "B365CD" in df.columns else ("B365D" if "B365D" in df.columns else "PSD")
        acol = "B365CA" if "B365CA" in df.columns else ("B365A" if "B365A" in df.columns else "PSA")
        if hcol in df.columns:
            df["odds_home"] = df[hcol]
            df["odds_draw"] = df[dcol]
            df["odds_away"] = df[acol]
            return df.dropna(subset=["match_date", "odds_home", "odds_draw", "odds_away"])
    return pd.DataFrame()


def _join_statsbomb_enrichment(matches: pd.DataFrame) -> pd.DataFrame:
    """Left-join StatsBomb event aggregates onto international matches."""
    stats = load_statsbomb_match_stats()
    if stats.empty:
        return matches

    stat_cols = ["shots", "shots_on_target", "xg", "corners", "fouls", "possession_pct"]
    stats = stats.copy()
    stats["match_date_norm"] = pd.to_datetime(stats["match_date"]).dt.normalize()

    home_stats = stats[stats["is_home"] == True].copy()  # noqa: E712
    home_stats = home_stats.rename(columns={"team": "home_team", "opponent": "away_team"})
    home_stats = home_stats.rename(columns={c: f"home_{c}" for c in stat_cols})

    away_stats = stats[stats["is_home"] == False].copy()  # noqa: E712
    away_stats = away_stats.rename(columns={"team": "away_team", "opponent": "home_team"})
    away_stats = away_stats.rename(columns={c: f"away_{c}" for c in stat_cols})

    join_cols = ["match_date_norm", "home_team", "away_team"]
    m = matches.copy()
    m["match_date_norm"] = pd.to_datetime(m["match_date"]).dt.normalize()

    m = m.merge(
        home_stats[join_cols + [f"home_{c}" for c in stat_cols]].drop_duplicates(join_cols),
        on=join_cols,
        how="left",
    )
    m = m.merge(
        away_stats[join_cols + [f"away_{c}" for c in stat_cols]].drop_duplicates(join_cols),
        on=join_cols,
        how="left",
    )
    return m.drop(columns=["match_date_norm"], errors="ignore")


def load_all_matches() -> pd.DataFrame:
    """Primary match table: martj42 backbone with StatsBomb and odds enrichment."""
    matches = load_international_results()
    if matches.empty:
        raise FileNotFoundError(
            "No international results found. Run: python scripts/download_data.py --international-only"
        )

    matches = _join_statsbomb_enrichment(matches)

    odds = load_odds()
    if not odds.empty:
        odds_key = odds[["match_date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]].copy()
        odds_key["match_date"] = pd.to_datetime(odds_key["match_date"]).dt.normalize()
        matches["match_date_norm"] = pd.to_datetime(matches["match_date"]).dt.normalize()
        matches = matches.merge(
            odds_key.rename(columns={"match_date": "match_date_norm"}),
            on=["match_date_norm", "home_team", "away_team"],
            how="left",
        )
        matches = matches.drop(columns=["match_date_norm"], errors="ignore")

    from src.polymarket_client import join_polymarket_odds_to_matches
    matches = join_polymarket_odds_to_matches(matches)

    return matches


# Re-export for backward compatibility (canonical implementation in statsbomb_shots.py).
from src.statsbomb_shots import load_statsbomb_shots  # noqa: E402

