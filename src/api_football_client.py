"""API-Football client with disk cache and martj42 fallback."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    API_FOOTBALL_BASE,
    API_FOOTBALL_KEY,
    API_FOOTBALL_RATE_LIMIT_SEC,
    API_FOOTBALL_WC_LEAGUE_ID,
    API_FOOTBALL_WC_SEASON,
    PROJECT_ROOT,
)
from src.data_loader import load_international_results
from src.team_mapping import fuzzy_match_team, normalize_team_name

CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "api_football"
FIXTURES_PATH = CACHE_DIR / "fixtures_wc2026.parquet"
SYNC_META_PATH = CACHE_DIR / "last_sync.json"

_LAST_REQUEST = 0.0
_LAST_API_ERROR: str | None = None


def get_last_api_error() -> str | None:
    return _LAST_API_ERROR


def _throttle() -> None:
    global _LAST_REQUEST
    elapsed = time.time() - _LAST_REQUEST
    if elapsed < API_FOOTBALL_RATE_LIMIT_SEC:
        time.sleep(API_FOOTBALL_RATE_LIMIT_SEC - elapsed)
    _LAST_REQUEST = time.time()


def _api_get(endpoint: str, params: dict | None = None) -> dict | None:
    global _LAST_API_ERROR
    if not API_FOOTBALL_KEY:
        _LAST_API_ERROR = "API_FOOTBALL_KEY not set in .env"
        return None
    _throttle()
    url = f"{API_FOOTBALL_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(
            url,
            params=params or {},
            headers={"x-apisports-key": API_FOOTBALL_KEY},
            timeout=30,
        )
        if resp.status_code == 200:
            body = resp.json()
            errs = body.get("errors") or {}
            if errs:
                if isinstance(errs, dict):
                    _LAST_API_ERROR = "; ".join(
                        f"{k}: {v}" for k, v in errs.items() if v
                    ) or str(errs)
                else:
                    _LAST_API_ERROR = str(errs)
                return body if body.get("response") else None
            _LAST_API_ERROR = None
            return body
        _LAST_API_ERROR = f"HTTP {resp.status_code}"
    except requests.RequestException as e:
        _LAST_API_ERROR = str(e)
    return None


def _parse_stage(round_name: str | None) -> str:
    if not round_name:
        return "group"
    r = round_name.lower()
    if "final" in r and "semi" not in r and "quarter" not in r:
        return "final"
    if "semi" in r:
        return "semi_final"
    if "quarter" in r:
        return "quarter_final"
    if "round of 16" in r or "1/8" in r or "8th" in r:
        return "round_of_16"
    return "group"


def _normalize_fixture_row(item: dict, known_teams: list[str] | None = None) -> dict | None:
    teams = item.get("teams", {})
    fixture = item.get("fixture", {})
    league = item.get("league", {})
    goals = item.get("goals", {}) or {}
    score = item.get("score", {}) or {}

    home_raw = teams.get("home", {}).get("name", "")
    away_raw = teams.get("away", {}).get("name", "")
    if not home_raw or not away_raw:
        return None

    home = normalize_team_name(home_raw, "api_football")
    away = normalize_team_name(away_raw, "api_football")
    if known_teams:
        home = fuzzy_match_team(home, known_teams) or home
        away = fuzzy_match_team(away, known_teams) or away

    date_str = fixture.get("date", "")
    try:
        match_date = pd.to_datetime(date_str, utc=True).tz_convert(None)
    except (TypeError, ValueError):
        return None

    status = (fixture.get("status") or {}).get("short", "NS")
    finished = status in ("FT", "AET", "PEN")
    upcoming = status in ("NS", "TBD", "PST")

    return {
        "fixture_id": fixture.get("id"),
        "match_date": match_date,
        "home_team": home,
        "away_team": away,
        "home_score": goals.get("home"),
        "away_score": goals.get("away"),
        "status": status,
        "is_finished": finished,
        "is_upcoming": upcoming,
        "stage": _parse_stage(league.get("round")),
        "round": league.get("round"),
        "venue": (fixture.get("venue") or {}).get("name"),
        "source": "api_football",
    }


def fetch_wc_fixtures_from_api(known_teams: list[str] | None = None) -> pd.DataFrame:
    """Fetch FIFA World Cup fixtures from API-Football."""
    body = _api_get(
        "fixtures",
        {"league": API_FOOTBALL_WC_LEAGUE_ID, "season": API_FOOTBALL_WC_SEASON},
    )
    if not body:
        return pd.DataFrame()

    rows = []
    for item in body.get("response", []):
        row = _normalize_fixture_row(item, known_teams)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_wc_fixtures_from_martj42() -> pd.DataFrame:
    """Fallback: WC 2026 fixtures from martj42 international results."""
    m = load_international_results()
    wc = m[
        m["tournament"].astype(str).str.fullmatch("FIFA World Cup", case=False, na=False)
        & (pd.to_datetime(m["match_date"]).dt.year == API_FOOTBALL_WC_SEASON)
    ].copy()

    if wc.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now().normalize()
    rows = []
    for _, r in wc.iterrows():
        md = pd.Timestamp(r["match_date"])
        finished = md < today
        rows.append({
            "fixture_id": r.get("match_id"),
            "match_date": md,
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_score": r["home_score"],
            "away_score": r["away_score"],
            "status": "FT" if finished else "NS",
            "is_finished": finished,
            "is_upcoming": not finished,
            "stage": "group",
            "round": "Group Stage",
            "venue": r.get("city"),
            "source": "martj42",
        })
    return pd.DataFrame(rows)


def sync_wc_fixtures(force: bool = False, known_teams: list[str] | None = None) -> pd.DataFrame:
    """Download and cache WC fixtures (API-Football or martj42 fallback)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if FIXTURES_PATH.exists() and not force:
        return load_wc_fixtures()

    df = pd.DataFrame()
    source = "none"

    if API_FOOTBALL_KEY:
        print("  Fetching fixtures from API-Football...")
        df = fetch_wc_fixtures_from_api(known_teams)
        if not df.empty:
            source = "api_football"

    if df.empty:
        print("  API-Football unavailable — using martj42 WC 2026 fallback...")
        df = fetch_wc_fixtures_from_martj42()
        source = "martj42"

    if not df.empty:
        df = df.sort_values("match_date").drop_duplicates(
            subset=["match_date", "home_team", "away_team"], keep="last"
        )
        df.to_parquet(FIXTURES_PATH, index=False)
        SYNC_META_PATH.write_text(
            json.dumps({
                "synced_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "count": len(df),
                "api_error": get_last_api_error() if source != "api_football" else None,
            }, indent=2),
            encoding="utf-8",
        )
    return df


def load_wc_fixtures() -> pd.DataFrame:
    """Load cached WC fixtures, syncing if missing."""
    if FIXTURES_PATH.exists():
        df = pd.read_parquet(FIXTURES_PATH)
        df["match_date"] = pd.to_datetime(df["match_date"])
        return df.sort_values("match_date").reset_index(drop=True)

    return sync_wc_fixtures()


def get_fixture_sync_meta() -> dict:
    if SYNC_META_PATH.exists():
        return json.loads(SYNC_META_PATH.read_text(encoding="utf-8"))
    return {}


def fetch_live_odds_for_fixture(fixture_id: int) -> dict | None:
    """Fetch live pre-match/match-winner odds for a fixture from API-Football, caching results for 1 hour."""
    if not fixture_id:
        return None

    cache_file = CACHE_DIR / f"odds_{fixture_id}.json"
    
    # Check if cache is valid (less than 1 hour old)
    if cache_file.exists():
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime < 3600:
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    if not API_FOOTBALL_KEY:
        return None

    print(f"  Fetching live odds for fixture {fixture_id} from API-Football...")
    body = _api_get("odds", {"fixture": fixture_id})
    if not body or not body.get("response"):
        return None

    # Cache the raw response on disk
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(body, indent=2), encoding="utf-8")
    except Exception:
        pass

    return body


def parse_match_winner_odds(odds_body: dict, home_team: str = "", away_team: str = "") -> dict | None:
    """Parse decimal match winner (1X2) odds from API-Football response. Returns decimal odds or None."""
    if not odds_body or "response" not in odds_body:
        return None

    response_list = odds_body.get("response", [])
    if not response_list:
        return None

    # Inspect bookmakers in the first response item
    item = response_list[0]
    bookmakers = item.get("bookmakers", [])
    if not bookmakers:
        return None

    home_lower = home_team.lower()
    away_lower = away_team.lower()

    # Look for "Match Winner" bet (id: 1) in bookmakers
    for bm in bookmakers:
        for bet in bm.get("bets", []):
            bet_id = bet.get("id")
            bet_name = bet.get("name", "")
            if bet_id == 1 or bet_name.lower() == "match winner":
                values = bet.get("values", [])
                if len(values) < 3:
                    continue

                home_odd = draw_odd = away_odd = None
                for val in values:
                    v_label = str(val.get("value", "")).lower()
                    v_odd = float(val.get("odd", 0.0))
                    
                    if v_label in ("home", "1") or (home_lower and home_lower in v_label):
                        home_odd = v_odd
                    elif v_label in ("draw", "x", "n", "tie"):
                        draw_odd = v_odd
                    elif v_label in ("away", "2") or (away_lower and away_lower in v_label):
                        away_odd = v_odd

                if home_odd and draw_odd and away_odd:
                    return {
                        "bookmaker": bm.get("name", "Unknown"),
                        "home_odds": home_odd,
                        "draw_odds": draw_odd,
                        "away_odds": away_odd
                    }

    return None
