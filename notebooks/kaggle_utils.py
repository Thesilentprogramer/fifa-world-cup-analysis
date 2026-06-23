"""
kaggle_utils.py — self-contained helper for the FIFA WC 2026 Kaggle notebook.

Includes:
  - Team name normalisation
  - ELO rating engine
  - Rolling feature engineering
  - H2H features
  - Penalty shootout Monte Carlo simulator
  - WC 2026 group fixtures (placeholder)

No live API keys, no Streamlit, no local file-system dependencies.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Team name normalisation
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    "Korea Republic": "South Korea", "Republic of Korea": "South Korea",
    "Korea, South": "South Korea", "Korea DPR": "North Korea",
    "IR Iran": "Iran", "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Cote d Ivoire": "Ivory Coast", "Czech Republic": "Czechia",
    "FYR Macedonia": "North Macedonia", "Macedonia": "North Macedonia",
    "Congo DR": "DR Congo", "Democratic Republic of Congo": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Türkiye": "Turkey", "Turkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Republic of Ireland": "Ireland", "China PR": "China",
    "Cape Verde Islands": "Cape Verde", "Trinidad & Tobago": "Trinidad and Tobago",
    "UAE": "United Arab Emirates", "USSR": "Russia",
    "West Germany": "Germany", "East Germany": "Germany",
    "Serbia and Montenegro": "Serbia", "Yugoslavia": "Serbia",
}


def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    return _ALIASES.get(name.strip(), name.strip())


# ---------------------------------------------------------------------------
# 2. ELO rating engine
# ---------------------------------------------------------------------------
K_FACTOR = 20
HOME_ADVANTAGE = 100
INITIAL_ELO = 1500.0


def compute_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-match ELO ratings.
    Input: DataFrame with [match_date, home_team, away_team, home_score, away_score].
    Returns same DataFrame with home_elo_pre and away_elo_pre columns added.
    """
    df = matches.sort_values("match_date").reset_index(drop=True).copy()
    ratings: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        h_elo = ratings.get(h, INITIAL_ELO)
        a_elo = ratings.get(a, INITIAL_ELO)
        home_elos.append(h_elo)
        away_elos.append(a_elo)

        h_adj = h_elo + HOME_ADVANTAGE
        exp_h = 1.0 / (1.0 + 10 ** ((a_elo - h_adj) / 400))
        exp_a = 1 - exp_h

        hs, aws = int(row["home_score"]), int(row["away_score"])
        act_h = 1.0 if hs > aws else (0.0 if hs < aws else 0.5)
        act_a = 1.0 - act_h

        ratings[h] = h_elo + K_FACTOR * (act_h - exp_h)
        ratings[a] = a_elo + K_FACTOR * (act_a - exp_a)

    df["home_elo_pre"] = home_elos
    df["away_elo_pre"] = away_elos
    return df


def get_elo_history(matches_elo: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Return a long-format ELO time-series for a list of teams."""
    records = []
    ratings: dict[str, float] = {}
    df = matches_elo.sort_values("match_date")
    for _, row in df.iterrows():
        for side, opp_side in [("home", "away"), ("away", "home")]:
            team = row[f"{side}_team"]
            if team in teams:
                records.append({
                    "date": row["match_date"],
                    "team": team,
                    "elo": row[f"{side}_elo_pre"],
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
ROLLING_WINDOWS = [5, 10]
H2H_WINDOW = 10

TOURNAMENT_KEYWORDS = {
    "world_cup": ["world cup"],
    "euro": ["european championship", "euro "],
    "qualifier": ["qualif"],
    "friendly": ["friendly"],
}

CONFEDERATION_MAP: dict[str, str] = {
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Peru": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Venezuela": "CONMEBOL",
    "Germany": "UEFA", "France": "UEFA", "Spain": "UEFA", "Italy": "UEFA",
    "England": "UEFA", "Netherlands": "UEFA", "Portugal": "UEFA",
    "Belgium": "UEFA", "Croatia": "UEFA", "Switzerland": "UEFA",
    "Poland": "UEFA", "Denmark": "UEFA", "Sweden": "UEFA",
    "Wales": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Austria": "UEFA", "Turkey": "UEFA", "Hungary": "UEFA",
    "Ukraine": "UEFA", "Czechia": "UEFA", "Slovakia": "UEFA",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC",
    "Iran": "AFC", "Saudi Arabia": "AFC", "Qatar": "AFC",
    "China": "AFC", "Iraq": "AFC", "Uzbekistan": "AFC",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF", "Panama": "CONCACAF",
    "Honduras": "CONCACAF", "El Salvador": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Nigeria": "CAF", "Senegal": "CAF", "Morocco": "CAF", "Ghana": "CAF",
    "Cameroon": "CAF", "Egypt": "CAF", "Tunisia": "CAF", "Ivory Coast": "CAF",
    "Algeria": "CAF", "South Africa": "CAF", "Mali": "CAF", "DR Congo": "CAF",
    "New Zealand": "OFC",
}


def _outcome(gf: int, ga: int) -> int:
    return 2 if gf > ga else (1 if gf == ga else 0)


def _points(gf: int, ga: int) -> int:
    return 3 if gf > ga else (1 if gf == ga else 0)


def _tournament_category(tournament: str) -> str:
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:
        return "world_cup"
    if ("euro" in t or "european championship" in t) and "qualif" not in t:
        return "euro"
    if "qualif" in t:
        return "qualifier"
    if "friendly" in t:
        return "friendly"
    return "other"


def build_match_base(matches: pd.DataFrame) -> pd.DataFrame:
    """Create two perspective rows per match."""
    rows = []
    for _, m in matches.iterrows():
        neutral = int(m.get("neutral", False))
        tcat = _tournament_category(str(m.get("tournament", "")))
        for team, opp, gf, ga, is_home in [
            (m["home_team"], m["away_team"], m["home_score"], m["away_score"], 1 if not neutral else 0),
            (m["away_team"], m["home_team"], m["away_score"], m["home_score"], 0),
        ]:
            rows.append({
                "match_id": m["match_id"],
                "match_date": m["match_date"],
                "tournament": m.get("tournament", ""),
                "tournament_category": tcat,
                "team": team, "opponent": opp,
                "goals_for": int(gf), "goals_against": int(ga),
                "is_home": is_home, "is_neutral": neutral,
                "stage": m.get("stage", "group"),
                "is_knockout": int(m.get("stage", "group") != "group"),
                "outcome": _outcome(int(gf), int(ga)),
                "points": _points(int(gf), int(ga)),
            })
    return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    merged = df.sort_values(["team", "match_date"]).copy()
    grp = merged.groupby("team", group_keys=False)
    for w in ROLLING_WINDOWS:
        merged[f"form_{w}"] = grp["points"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).sum()
        )
        merged[f"goals_for_avg_{w}"] = grp["goals_for"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        merged[f"goals_against_avg_{w}"] = grp["goals_against"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
    return merged


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["team", "match_date"]).copy()
    result["team_rest_days"] = (
        result.groupby("team")["match_date"].diff().dt.days.fillna(14).clip(1, 365)
    )
    opp_rest = (
        result[["match_id", "team", "team_rest_days"]]
        .drop_duplicates(subset=["match_id", "team"])
        .rename(columns={"team": "opponent", "team_rest_days": "opp_rest_days"})
    )
    result = result.merge(opp_rest, on=["match_id", "opponent"], how="left")
    result["opp_rest_days"] = result["opp_rest_days"].fillna(14)
    result["rest_days_diff"] = result["team_rest_days"] - result["opp_rest_days"]
    return result


def compute_tournament_form(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["team", "tournament_category", "match_date"]).copy()
    grp = result.groupby(["team", "tournament_category"], group_keys=False)
    result["tournament_form_5"] = grp["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    result["tournament_gf_avg_5"] = grp["goals_for"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    result["tournament_ga_avg_5"] = grp["goals_against"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    return result


def compute_h2h_features(base: pd.DataFrame) -> pd.DataFrame:
    pair_hist: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    h2h_lookup: dict = {}

    def _stats(prior):
        if not prior:
            return {"h2h_n": 0, "h2h_wins": 0, "h2h_draws": 0, "h2h_losses": 0,
                    "h2h_gf": 0.0, "h2h_ga": 0.0}
        return {
            "h2h_n": len(prior),
            "h2h_wins": sum(p["outcome"] == 2 for p in prior),
            "h2h_draws": sum(p["outcome"] == 1 for p in prior),
            "h2h_losses": sum(p["outcome"] == 0 for p in prior),
            "h2h_gf": float(sum(p["gf"] for p in prior)),
            "h2h_ga": float(sum(p["ga"] for p in prior)),
        }

    for _, row in base.sort_values("match_date").iterrows():
        key = (row["match_id"], row["team"], row["opponent"])
        if key not in h2h_lookup:
            prior = list(pair_hist[(row["team"], row["opponent"])])
            h2h_lookup[key] = _stats(prior)
            pair_hist[(row["team"], row["opponent"])].append({
                "gf": row["goals_for"], "ga": row["goals_against"],
                "outcome": row["outcome"],
            })
            pair_hist[(row["opponent"], row["team"])].append({
                "gf": row["goals_against"], "ga": row["goals_for"],
                "outcome": 2 if row["outcome"] == 0 else (0 if row["outcome"] == 2 else 1),
            })

    records = [h2h_lookup.get((r["match_id"], r["team"], r["opponent"]), _stats([])) for _, r in base.iterrows()]
    return pd.concat([base.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def add_elo_perspective(df: pd.DataFrame, matches_elo: pd.DataFrame) -> pd.DataFrame:
    elo_map = matches_elo.set_index("match_id")[["home_team", "away_team", "home_elo_pre", "away_elo_pre"]]
    result = df.copy()

    def _team_elo(row):
        m = elo_map.loc[row["match_id"]]
        return m["home_elo_pre"] if row["team"] == m["home_team"] else m["away_elo_pre"]

    def _opp_elo(row):
        m = elo_map.loc[row["match_id"]]
        return m["away_elo_pre"] if row["team"] == m["home_team"] else m["home_elo_pre"]

    result["team_elo"] = result.apply(_team_elo, axis=1)
    result["opponent_elo"] = result.apply(_opp_elo, axis=1)
    result["elo_diff"] = result["team_elo"] - result["opponent_elo"]
    return result


def add_confederation_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["confederation"] = result["team"].map(CONFEDERATION_MAP).fillna("OTHER")
    result["opp_confederation"] = result["opponent"].map(CONFEDERATION_MAP).fillna("OTHER")
    result["same_confederation"] = (result["confederation"] == result["opp_confederation"]).astype(int)
    return result


def add_form_diff(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for w in ROLLING_WINDOWS:
        col = f"form_{w}"
        if col not in result.columns:
            continue
        opp = (df[["match_id", "team", col]]
               .drop_duplicates(subset=["match_id", "team"])
               .rename(columns={"team": "opponent", col: f"opp_{col}"}))
        result = result.merge(opp, on=["match_id", "opponent"], how="left")
        result[f"form_diff_{w}"] = result[col] - result[f"opp_{col}"]
        result = result.drop(columns=[f"opp_{col}"])
    return result


FEATURE_COLUMNS = [
    "is_home", "is_neutral", "is_knockout", "same_confederation",
    "team_elo", "opponent_elo", "elo_diff",
    "form_5", "form_10", "form_diff_5", "form_diff_10",
    "goals_for_avg_5", "goals_against_avg_5",
    "goals_for_avg_10", "goals_against_avg_10",
    "tournament_form_5", "tournament_gf_avg_5", "tournament_ga_avg_5",
    "team_rest_days", "rest_days_diff",
    "h2h_n", "h2h_wins", "h2h_draws", "h2h_losses",
    "h2h_gf", "h2h_ga",
]


def build_full_features(matches: pd.DataFrame) -> pd.DataFrame:
    """End-to-end feature pipeline. Input: raw matches DataFrame."""
    matches = matches.copy()
    matches["home_team"] = matches["home_team"].apply(normalize_team)
    matches["away_team"] = matches["away_team"].apply(normalize_team)

    matches_elo = compute_elo_ratings(
        matches[["match_id", "match_date", "home_team", "away_team",
                 "home_score", "away_score"]].drop_duplicates("match_id")
    )
    base = build_match_base(matches)
    df = compute_rolling_features(base)
    df = compute_rest_days(df)
    df = compute_tournament_form(df)
    df = compute_h2h_features(df)
    df = add_elo_perspective(df, matches_elo)
    df = add_confederation_features(df)
    df = add_form_diff(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    return df, matches_elo


# ---------------------------------------------------------------------------
# 4. Penalty shootout Monte Carlo
# ---------------------------------------------------------------------------
BASE_CONVERSION = 0.757
PRESSURE_DECAY = 0.03
FIRST_KICK_ADVANTAGE = 0.012

KEEPER_PRESETS = {
    "Average":             0.00,
    "Emi Martínez":        0.12,
    "Dominik Livaković":   0.11,
    "Yann Sommer":         0.10,
    "Thibaut Courtois":    0.09,
    "Manuel Neuer":        0.08,
    "Jordan Pickford":     0.10,
    "Hugo Lloris":         0.07,
    "Ederson":             0.06,
}


def _conv_prob(taker_skill: float, keeper_save: float, round_num: int, first: bool) -> float:
    p = BASE_CONVERSION + taker_skill - keeper_save
    if round_num > 5:
        p -= PRESSURE_DECAY * (round_num - 5)
    if first:
        p += FIRST_KICK_ADVANTAGE
    return float(np.clip(p, 0.30, 0.98))


def simulate_shootout(
    team_a_skills: list[float],
    team_b_skills: list[float],
    keeper_a_save: float = 0.0,
    keeper_b_save: float = 0.0,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    wins_a = wins_b = 0
    total_rounds = 0
    sample_log: list[dict] = []

    for i in range(n_sims):
        first = "a" if i % 2 == 0 else "b"
        sa, sb = 0, 0
        log: list[dict] = []
        for r in range(1, 31):
            for kick_first in [True, False]:
                if kick_first:
                    team = first
                else:
                    team = "b" if first == "a" else "a"
                skills = team_a_skills if team == "a" else team_b_skills
                keeper_save = keeper_b_save if team == "a" else keeper_a_save
                skill = skills[(r - 1) % len(skills)]
                prob = _conv_prob(skill, keeper_save, r, kick_first)
                scored = bool(rng.random() < prob)
                if team == "a" and scored:
                    sa += 1
                elif team == "b" and scored:
                    sb += 1
                log.append({"round": r, "team": team, "scored": scored,
                             "score_a": sa, "score_b": sb})
                if r <= 5:
                    rem_a = 5 - r if (kick_first and team == "a") or (not kick_first and team == "b") else 5 - r + 1
                    # simplified early termination
                if r > 5 and not kick_first:
                    if sa != sb:
                        break
            if r > 5 and sa != sb:
                break
            if r <= 5:
                rem = 5 - r
                if sa > sb + rem or sb > sa + rem:
                    break
        winner = "a" if sa > sb else "b"
        if winner == "a":
            wins_a += 1
        else:
            wins_b += 1
        total_rounds += r
        if i == 0:
            sample_log = log

    return {
        "win_prob_a": wins_a / n_sims,
        "win_prob_b": wins_b / n_sims,
        "avg_rounds": total_rounds / n_sims,
        "sample_log": sample_log,
    }


# ---------------------------------------------------------------------------
# 5. WC 2026 confirmed + likely qualified teams (placeholder groups)
# ---------------------------------------------------------------------------
WC2026_LIKELY_TEAMS = [
    # UEFA (16 teams)
    "Germany", "Spain", "France", "England", "Portugal", "Netherlands",
    "Belgium", "Italy", "Croatia", "Denmark", "Austria", "Switzerland",
    "Serbia", "Poland", "Turkey", "Ukraine",
    # CONMEBOL (6 teams)
    "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Chile",
    # CONCACAF (6 teams)
    "United States", "Mexico", "Canada", "Panama", "Costa Rica", "Jamaica",
    # CAF (9 teams)
    "Morocco", "Senegal", "Nigeria", "Ivory Coast", "Egypt",
    "Cameroon", "Ghana", "Algeria", "Tunisia",
    # AFC (8 teams)
    "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
    "Qatar", "Iraq", "Uzbekistan",
    # OFC (1)
    "New Zealand",
    # Host nations (already counted above)
]

# Note: Actual WC 2026 draw will assign 48 teams to 12 groups of 4.
# Use as placeholder until official draw.
WC2026_SAMPLE_GROUPS = {
    "A": ["United States", "Morocco", "Uruguay", "Serbia"],
    "B": ["Mexico", "Germany", "Japan", "Ivory Coast"],
    "C": ["Canada", "France", "Argentina", "Cameroon"],
    "D": ["Spain", "Brazil", "South Korea", "Australia"],
    "E": ["England", "Netherlands", "Colombia", "Senegal"],
    "F": ["Portugal", "Belgium", "Mexico", "Ghana"],
    "G": ["Italy", "Croatia", "Ecuador", "Nigeria"],
    "H": ["Denmark", "Switzerland", "Saudi Arabia", "Algeria"],
    "I": ["Austria", "Poland", "Iran", "Tunisia"],
    "J": ["Turkey", "Ukraine", "Qatar", "New Zealand"],
    "K": ["Chile", "Panama", "Iraq", "Egypt"],
    "L": ["Costa Rica", "Jamaica", "Uzbekistan", "Paraguay"],
}
