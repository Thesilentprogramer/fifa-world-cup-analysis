"""Shot-level StatsBomb event loader (isolated to avoid import cycles)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.team_mapping import normalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATSBOMB_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb"


def _nested_name(val: Any) -> str:
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


def _parse_events_file(path: Path) -> list:
    try:
        text = path.read_text(encoding="utf-8").replace("NaN", "null")
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return []


def load_statsbomb_shots(data_dir: Path | None = None) -> pd.DataFrame:
    """Load shot-level rows from cached StatsBomb events for xG model training."""
    base = data_dir or STATSBOMB_DIR
    rows: list[dict[str, Any]] = []

    if not base.exists():
        return pd.DataFrame()

    goal_center = (120.0, 40.0)

    for comp_dir in sorted(base.iterdir()):
        if not comp_dir.is_dir():
            continue
        for season_dir in sorted(comp_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            events_dir = season_dir / "events_by_match"
            if not events_dir.exists():
                continue

            for match_events_file in sorted(events_dir.glob("*.json")):
                events = _parse_events_file(match_events_file)
                for e in events:
                    if _event_type_name(e) != "Shot" and not e.get("shot_statsbomb_xg"):
                        continue
                    loc = e.get("location")
                    if not loc or not isinstance(loc, (list, tuple)) or len(loc) < 2:
                        continue
                    x, y = float(loc[0]), float(loc[1])
                    dx = goal_center[0] - x
                    dy = goal_center[1] - y
                    distance = math.hypot(dx, dy)
                    angle = math.degrees(2 * math.atan2(abs(dy), max(dx, 0.1)))

                    outcome = _event_shot_outcome(e)
                    is_goal = int(outcome == "Goal")
                    team = normalize_team_name(_event_team_name(e)) if _event_team_name(e) else ""

                    rows.append({
                        "match_id": e.get("match_id"),
                        "minute": e.get("minute", 0),
                        "team": team,
                        "x": x,
                        "y": y,
                        "distance": distance,
                        "angle": angle,
                        "body_part": e.get("shot_body_part") or "Foot",
                        "technique": e.get("shot_technique") or "Normal",
                        "shot_type": e.get("shot_type") or "Open Play",
                        "is_goal": is_goal,
                        "statsbomb_xg": _event_shot_xg(e),
                    })

    return pd.DataFrame(rows)
