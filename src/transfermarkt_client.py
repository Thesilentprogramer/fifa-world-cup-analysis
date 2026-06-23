"""Transfermarkt API client with disk cache (felipeall/transfermarkt-api)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import PROJECT_ROOT, TRANSFERMARKT_API_BASE, TRANSFERMARKT_RATE_LIMIT_SEC
from src.team_mapping import normalize_team_name

CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
TEAM_IDS_PATH = CACHE_DIR / "national_team_ids.json"
SQUAD_VALUES_PATH = CACHE_DIR / "squad_values_by_year.csv"

_YOUTH_PATTERN = re.compile(r"\bU\d{1,2}\b", re.IGNORECASE)
_LAST_REQUEST = 0.0


def _throttle() -> None:
    global _LAST_REQUEST
    elapsed = time.time() - _LAST_REQUEST
    if elapsed < TRANSFERMARKT_RATE_LIMIT_SEC:
        time.sleep(TRANSFERMARKT_RATE_LIMIT_SEC - elapsed)
    _LAST_REQUEST = time.time()


def _get(path: str, params: dict | None = None) -> dict | list | None:
    _throttle()
    url = f"{TRANSFERMARKT_API_BASE.rstrip('/')}/{path.lstrip('/')}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def _load_team_ids() -> dict[str, str]:
    if TEAM_IDS_PATH.exists():
        with open(TEAM_IDS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_team_ids(mapping: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEAM_IDS_PATH.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")


def _is_national_team_result(result: dict, team_name: str) -> bool:
    name = result.get("name", "")
    if _YOUTH_PATTERN.search(name):
        return False
    country = result.get("country", "")
    # Prefer exact country match (e.g. "Brazil" not "Mamelodi Sundowns FC")
    if country and country.lower() == team_name.lower():
        return True
    if name.lower() == team_name.lower():
        return True
    return False


def search_national_team_id(team: str) -> str | None:
    """Resolve canonical team name to Transfermarkt club id."""
    team = normalize_team_name(team, "international")
    mapping = _load_team_ids()
    if team in mapping:
        return mapping[team]

    data = _get(f"clubs/search/{team}")
    if not isinstance(data, dict):
        return None

    for result in data.get("results", []):
        if _is_national_team_result(result, team):
            club_id = str(result["id"])
            mapping[team] = club_id
            _save_team_ids(mapping)
            return club_id

    # Fallback: first result with matching country
    for result in data.get("results", []):
        if result.get("country", "").lower() == team.lower() and not _YOUTH_PATTERN.search(result.get("name", "")):
            club_id = str(result["id"])
            mapping[team] = club_id
            _save_team_ids(mapping)
            return club_id
    return None


def fetch_squad_market_value(team: str, year: int | None = None) -> float | None:
    """Fetch squad market value (EUR) from club search or cached values."""
    team = normalize_team_name(team, "international")
    club_id = search_national_team_id(team)
    if not club_id:
        return None

    data = _get(f"clubs/search/{team}")
    if isinstance(data, dict):
        for result in data.get("results", []):
            if str(result.get("id")) == club_id:
                mv = result.get("marketValue")
                if mv is not None:
                    return float(mv)
    return None


def build_squad_values_table(teams: list[str], years: list[int] | None = None) -> pd.DataFrame:
    """Build or extend squad values cache for teams."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame()
    if SQUAD_VALUES_PATH.exists():
        existing = pd.read_csv(SQUAD_VALUES_PATH)

    if years is None:
        years = list(range(2016, 2027))

    records: list[dict] = []
    for team in sorted(set(teams)):
        team = normalize_team_name(team, "international")
        mv = fetch_squad_market_value(team)
        if mv is None:
            continue
        for year in years:
            records.append({
                "team": team,
                "tournament_year": year,
                "squad_market_value": mv,
            })

    if not records and existing.empty:
        return pd.DataFrame(columns=["team", "tournament_year", "squad_market_value"])

    new_df = pd.DataFrame(records)
    if not existing.empty:
        combined = pd.concat([existing, new_df]).drop_duplicates(
            subset=["team", "tournament_year"], keep="last"
        )
    else:
        combined = new_df.drop_duplicates(subset=["team", "tournament_year"])

    combined.to_csv(SQUAD_VALUES_PATH, index=False)
    return combined


def load_squad_values() -> pd.DataFrame:
    """Load cached squad values table."""
    if not SQUAD_VALUES_PATH.exists():
        return pd.DataFrame(columns=["team", "tournament_year", "squad_market_value"])
    return pd.read_csv(SQUAD_VALUES_PATH)


def join_squad_values_to_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Join squad market value features to perspective rows."""
    stats = load_squad_values()
    result = df.copy()
    if "tournament_year" not in result.columns:
        result["tournament_year"] = pd.to_datetime(result["match_date"]).dt.year

    for col in ("squad_market_value", "opponent_squad_market_value", "squad_value_diff"):
        if col in result.columns:
            result = result.drop(columns=[col])

    if stats.empty:
        result["squad_market_value"] = float("nan")
        result["opponent_squad_market_value"] = float("nan")
        result["squad_value_diff"] = float("nan")
        return result

    merged = result.merge(
        stats[["team", "tournament_year", "squad_market_value"]],
        on=["team", "tournament_year"],
        how="left",
    )
    opp = stats.rename(columns={
        "team": "opponent",
        "squad_market_value": "opponent_squad_market_value",
    })
    merged = merged.merge(
        opp[["opponent", "tournament_year", "opponent_squad_market_value"]],
        on=["opponent", "tournament_year"],
        how="left",
    )
    merged["squad_value_diff"] = merged["squad_market_value"] - merged["opponent_squad_market_value"]

    merged["squad_market_value"] = merged.groupby("team")["squad_market_value"].transform(
        lambda s: s.ffill().bfill()
    )
    merged["opponent_squad_market_value"] = merged.groupby("opponent")["opponent_squad_market_value"].transform(
        lambda s: s.ffill().bfill()
    )
    merged["squad_value_diff"] = merged["squad_market_value"] - merged["opponent_squad_market_value"]
    return merged
