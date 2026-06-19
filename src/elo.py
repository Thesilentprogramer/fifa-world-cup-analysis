"""Elo rating computation for international football matches."""

from __future__ import annotations

import numpy as np
import pandas as pd

K_FACTOR = 20
HOME_ADVANTAGE = 100
INITIAL_ELO = 1500.0


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def compute_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-match Elo for each team in each match row.
    Input: one row per match with home_team, away_team, home_score, away_score, match_date.
    Returns matches with home_elo_pre, away_elo_pre columns.
    """
    df = matches.sort_values("match_date").reset_index(drop=True).copy()
    ratings: dict[str, float] = {}

    home_elos: list[float] = []
    away_elos: list[float] = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_elo = ratings.get(home, INITIAL_ELO)
        away_elo = ratings.get(away, INITIAL_ELO)
        home_elos.append(home_elo)
        away_elos.append(away_elo)

        home_adj = home_elo + HOME_ADVANTAGE
        exp_home = _expected_score(home_adj, away_elo)
        exp_away = 1 - exp_home

        hs, aws = int(row["home_score"]), int(row["away_score"])
        if hs > aws:
            act_home, act_away = 1.0, 0.0
        elif hs < aws:
            act_home, act_away = 0.0, 1.0
        else:
            act_home, act_away = 0.5, 0.5

        ratings[home] = home_elo + K_FACTOR * (act_home - exp_home)
        ratings[away] = away_elo + K_FACTOR * (act_away - exp_away)

    df["home_elo_pre"] = home_elos
    df["away_elo_pre"] = away_elos
    return df


def add_elo_to_perspective_rows(df: pd.DataFrame, matches_elo: pd.DataFrame) -> pd.DataFrame:
    """Join pre-match Elo onto team-perspective feature rows."""
    elo_map = matches_elo.set_index("match_id")[["home_team", "away_team", "home_elo_pre", "away_elo_pre"]]

    def team_elo(row):
        m = elo_map.loc[row["match_id"]]
        if row["team"] == m["home_team"]:
            return m["home_elo_pre"]
        return m["away_elo_pre"]

    def opp_elo(row):
        m = elo_map.loc[row["match_id"]]
        if row["team"] == m["home_team"]:
            return m["away_elo_pre"]
        return m["home_elo_pre"]

    result = df.copy()
    result["team_elo"] = result.apply(team_elo, axis=1)
    result["opponent_elo"] = result.apply(opp_elo, axis=1)
    result["elo_diff"] = result["team_elo"] - result["opponent_elo"]
    return result
