"""Match-level feature engineering for outcome prediction."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import load_all_matches, tournament_category
from src.elo import add_elo_to_perspective_rows, compute_elo_ratings
from src.odds_comparison import odds_to_implied_probs
from src.player_stats import join_player_stats_to_matches
from src.polymarket_client import add_polymarket_features
from src.transfermarkt_client import join_squad_values_to_matches
from src.team_mapping import normalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ROLLING_WINDOWS = [5, 10]
H2H_WINDOW = 10

CONFEDERATION_MAP: dict[str, str] = {
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Peru": "CONMEBOL",
    "Germany": "UEFA", "France": "UEFA", "Spain": "UEFA", "Italy": "UEFA",
    "England": "UEFA", "Netherlands": "UEFA", "Portugal": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Poland": "UEFA", "Denmark": "UEFA",
    "Sweden": "UEFA", "Wales": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC", "Iran": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "China": "AFC",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF",
    "Nigeria": "CAF", "Senegal": "CAF", "Morocco": "CAF", "Ghana": "CAF",
    "Cameroon": "CAF", "Egypt": "CAF", "Tunisia": "CAF", "Ivory Coast": "CAF",
    "Algeria": "CAF", "South Africa": "CAF",
}


def _match_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _outcome_from_perspective(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 2
    if goals_for == goals_against:
        return 1
    return 0


def build_match_base(matches: pd.DataFrame) -> pd.DataFrame:
    """Create two rows per match (home and away perspective)."""
    rows = []
    stat_suffixes = ["shots", "shots_on_target", "xg", "corners", "fouls", "possession_pct"]

    for _, m in matches.iterrows():
        home_stats = {f"{s}": m.get(f"home_{s}", np.nan) for s in stat_suffixes}
        away_stats = {f"{s}": m.get(f"away_{s}", np.nan) for s in stat_suffixes}
        neutral = int(m.get("is_neutral", 0))
        tcat = m.get("tournament_category", tournament_category(m.get("tournament", "")))

        for team, opp, gf, ga, is_home, tstats in [
            (m["home_team"], m["away_team"], m["home_score"], m["away_score"], 1 if not neutral else 0, home_stats),
            (m["away_team"], m["home_team"], m["away_score"], m["home_score"], 0 if not neutral else 0, away_stats),
        ]:
            row = {
                "match_id": m["match_id"],
                "match_date": m["match_date"],
                "competition": m.get("competition", ""),
                "tournament": m.get("tournament", ""),
                "tournament_category": tcat,
                "team": team,
                "opponent": opp,
                "goals_for": gf,
                "goals_against": ga,
                "is_home": is_home,
                "is_neutral": neutral,
                "stage": m.get("stage", "group"),
                "is_knockout": int(m.get("stage", "group") not in ("group",)),
                "outcome": _outcome_from_perspective(gf, ga),
                "points": _match_points(gf, ga),
                "odds_home": m.get("odds_home", np.nan),
                "odds_draw": m.get("odds_draw", np.nan),
                "odds_away": m.get("odds_away", np.nan),
                "polymarket_prob_home": m.get("polymarket_prob_home", np.nan),
                "polymarket_prob_draw": m.get("polymarket_prob_draw", np.nan),
                "polymarket_prob_away": m.get("polymarket_prob_away", np.nan),
            }
            row.update(tstats)
            rows.append(row)

    return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous match."""
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


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling form and event aggregates."""
    merged = df.sort_values(["team", "match_date"]).copy()
    stat_cols = ["shots", "shots_on_target", "xg", "corners", "fouls", "possession_pct"]
    team_groups = merged.groupby("team", group_keys=False)

    for window in ROLLING_WINDOWS:
        merged[f"form_{window}"] = team_groups["points"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        merged[f"goals_for_avg_{window}"] = team_groups["goals_for"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        merged[f"goals_against_avg_{window}"] = team_groups["goals_against"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        for col in stat_cols:
            if col in merged.columns:
                merged[f"{col}_avg_{window}"] = team_groups[col].transform(
                    lambda s: s.shift(1).rolling(window, min_periods=1).mean()
                )

    return merged


def compute_tournament_form(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling form within the same tournament category."""
    result = df.sort_values(["team", "tournament_category", "match_date"]).copy()
    grp = result.groupby(["team", "tournament_category"], group_keys=False)
    result["tournament_form_5"] = grp["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    result["tournament_goals_for_avg_5"] = grp["goals_for"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    result["tournament_goals_against_avg_5"] = grp["goals_against"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    return result


def compute_h2h_features(base: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head features using incremental pair history (one update per match)."""
    pair_hist: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    h2h_lookup: dict[tuple[str, str, str], dict] = {}  # (match_id, team, opp) -> stats

    home_rows = base[base["is_home"] == 1].sort_values("match_date")
    for _, row in home_rows.iterrows():
        team, opp, mid = row["team"], row["opponent"], row["match_id"]
        prior_team = list(pair_hist[(team, opp)])
        prior_opp = list(pair_hist[(opp, team)])

        h2h_lookup[(mid, team, opp)] = _h2h_stats(prior_team)
        h2h_lookup[(mid, opp, team)] = _h2h_stats(prior_opp)

        pair_hist[(team, opp)].append({
            "goals_for": row["goals_for"], "goals_against": row["goals_against"],
            "outcome": row["outcome"],
        })
        pair_hist[(opp, team)].append({
            "goals_for": row["goals_against"], "goals_against": row["goals_for"],
            "outcome": 2 if row["outcome"] == 0 else (0 if row["outcome"] == 2 else 1),
        })

    # Neutral-site matches: is_home may be 0 for both; handle remaining rows
    for _, row in base.sort_values("match_date").iterrows():
        key = (row["match_id"], row["team"], row["opponent"])
        if key not in h2h_lookup:
            prior = list(pair_hist[(row["team"], row["opponent"])])
            h2h_lookup[key] = _h2h_stats(prior)
            pair_hist[(row["team"], row["opponent"])].append({
                "goals_for": row["goals_for"], "goals_against": row["goals_against"],
                "outcome": row["outcome"],
            })
            pair_hist[(row["opponent"], row["team"])].append({
                "goals_for": row["goals_against"], "goals_against": row["goals_for"],
                "outcome": 2 if row["outcome"] == 0 else (0 if row["outcome"] == 2 else 1),
            })

    records = [
        h2h_lookup.get((r["match_id"], r["team"], r["opponent"]), _h2h_stats([]))
        for _, r in base.iterrows()
    ]
    return pd.concat([base.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def _h2h_stats(prior: list) -> dict:
    if not prior:
        return {
            "h2h_matches": 0, "h2h_wins": 0, "h2h_draws": 0, "h2h_losses": 0,
            "h2h_goals_for": 0, "h2h_goals_against": 0,
        }
    return {
        "h2h_matches": len(prior),
        "h2h_wins": sum(p["outcome"] == 2 for p in prior),
        "h2h_draws": sum(p["outcome"] == 1 for p in prior),
        "h2h_losses": sum(p["outcome"] == 0 for p in prior),
        "h2h_goals_for": sum(p["goals_for"] for p in prior),
        "h2h_goals_against": sum(p["goals_against"] for p in prior),
    }


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bookmaker odds to implied probability features from team perspective."""
    result = df.copy()

    def team_probs(row):
        if pd.isna(row.get("odds_home")):
            return pd.Series({"implied_prob_win": np.nan, "implied_prob_draw": np.nan, "implied_prob_loss": np.nan})
        probs = odds_to_implied_probs(row["odds_home"], row["odds_draw"], row["odds_away"])
        if row["is_home"] == 1:
            return pd.Series({"implied_prob_win": probs[0], "implied_prob_draw": probs[1], "implied_prob_loss": probs[2]})
        return pd.Series({"implied_prob_win": probs[2], "implied_prob_draw": probs[1], "implied_prob_loss": probs[0]})

    odds_feats = result.apply(team_probs, axis=1)
    result = pd.concat([result, odds_feats], axis=1)
    result["implied_prob_diff"] = result["implied_prob_win"] - result["implied_prob_loss"]
    return result


def add_opponent_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Difference features between team and opponent."""
    result = df.copy()
    opp_base = df[["match_id", "team"]].drop_duplicates(subset=["match_id", "team"]).rename(columns={"team": "opponent"})

    for window in ROLLING_WINDOWS:
        form_col = f"form_{window}"
        if form_col in result.columns:
            opp_vals = df[["match_id", "team", form_col]].drop_duplicates(subset=["match_id", "team"])
            opp_vals = opp_vals.rename(columns={"team": "opponent", form_col: f"opp_{form_col}"})
            result = result.merge(opp_vals, on=["match_id", "opponent"], how="left")
            result[f"form_diff_{window}"] = result[form_col] - result[f"opp_{form_col}"]
            result = result.drop(columns=[f"opp_{form_col}"])

        xg_col = f"xg_avg_{window}"
        if xg_col in result.columns:
            opp_vals = df[["match_id", "team", xg_col]].drop_duplicates(subset=["match_id", "team"])
            opp_vals = opp_vals.rename(columns={"team": "opponent", xg_col: f"opp_{xg_col}"})
            result = result.merge(opp_vals, on=["match_id", "opponent"], how="left")
            result[f"xg_diff_{window}"] = result[xg_col] - result[f"opp_{xg_col}"]
            result = result.drop(columns=[f"opp_{xg_col}"])

    result["confederation"] = result["team"].map(CONFEDERATION_MAP).fillna("OTHER")
    result["opp_confederation"] = result["opponent"].map(CONFEDERATION_MAP).fillna("OTHER")
    result["same_confederation"] = (result["confederation"] == result["opp_confederation"]).astype(int)
    return result


FEATURE_COLUMNS = [
    "is_home", "is_neutral", "is_knockout", "same_confederation",
    "team_elo", "opponent_elo", "elo_diff",
    "form_5", "form_10", "form_diff_5", "form_diff_10",
    "goals_for_avg_5", "goals_against_avg_5", "goals_for_avg_10", "goals_against_avg_10",
    "shots_avg_5", "shots_on_target_avg_5", "xg_avg_5", "xg_diff_5",
    "corners_avg_5", "fouls_avg_5", "possession_pct_avg_5",
    "tournament_form_5", "tournament_goals_for_avg_5", "tournament_goals_against_avg_5",
    "team_rest_days", "rest_days_diff",
    "implied_prob_win", "implied_prob_draw", "implied_prob_loss", "implied_prob_diff",
    "polymarket_prob_win", "polymarket_prob_draw", "polymarket_prob_loss", "polymarket_prob_diff",
    "squad_market_value", "squad_value_diff",
    "fbref_goals_per90", "fbref_xg_per90", "fbref_assists_per90",
    "h2h_matches", "h2h_wins", "h2h_draws", "h2h_losses",
    "h2h_goals_for", "h2h_goals_against",
]


def build_features(output_path: Path | None = None) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    matches = load_all_matches()
    matches_elo = compute_elo_ratings(
        matches[["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]].drop_duplicates("match_id")
    )

    base = build_match_base(matches)
    df = compute_rest_days(base)
    df = compute_rolling_features(df)
    df = compute_tournament_form(df)
    df = compute_h2h_features(df)
    df = add_elo_to_perspective_rows(df, matches_elo)
    df = add_odds_features(df)
    df = add_polymarket_features(df)
    df = join_player_stats_to_matches(df)
    df = join_squad_values_to_matches(df)
    df = add_opponent_adjusted_features(df)

    stage_map = {"group": 0, "round_of_16": 1, "quarter_final": 2, "semi_final": 3, "final": 4}
    df["stage_encoded"] = df["stage"].map(stage_map).fillna(0).astype(int)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    out = output_path or PROCESSED_DIR / "match_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def get_latest_team_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Most recent feature snapshot per team for inference."""
    latest = df.sort_values("match_date").groupby("team").tail(1).set_index("team")
    return {team: latest.loc[team] for team in latest.index}
