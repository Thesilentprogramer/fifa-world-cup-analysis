"""Monte Carlo simulation for FIFA World Cup group stage."""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
from src.match_predictor import MatchPredictor


def sample_match_score(outcome: str) -> tuple[int, int]:
    """Sample realistic goals for a match given outcome ('Win', 'Draw', 'Loss' for home team)."""
    if outcome == "Draw":
        # Draw probabilities: 1-1 is most common, then 0-0, 2-2, etc.
        r = random.random()
        if r < 0.30:
            return 0, 0
        elif r < 0.80:
            return 1, 1
        elif r < 0.97:
            return 2, 2
        else:
            return 3, 3
    
    # Decide goal difference (GD)
    r = random.random()
    if r < 0.60:
        gd = 1
    elif r < 0.85:
        gd = 2
    elif r < 0.95:
        gd = 3
    else:
        gd = 4

    # Opponent goals (Poisson-like distribution)
    opp_r = random.random()
    if opp_r < 0.55:
        opp_g = 0
    elif opp_r < 0.85:
        opp_g = 1
    elif opp_r < 0.96:
        opp_g = 2
    else:
        opp_g = 3

    if outcome == "Win":
        return opp_g + gd, opp_g
    else: # Loss
        return opp_g, opp_g + gd


def simulate_group_stage(
    teams: list[str],
    finished_matches: pd.DataFrame,
    upcoming_matches: pd.DataFrame,
    predictor: MatchPredictor,
    n_simulations: int = 2000,
) -> dict[str, dict[str, any]]:
    """
    Run Monte Carlo simulation of remaining group matches.
    Returns a dict containing simulation stats (avg pts, probabilities of ranks 1st-4th) for each team.
    """
    # 1. Compute baseline stats from already finished matches
    baseline_stats = {}
    for team in teams:
        baseline_stats[team] = {
            "played": 0, "won": 0, "drawn": 0, "lost": 0,
            "gf": 0, "ga": 0, "pts": 0
        }

    for _, row in finished_matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, aws = int(row["home_score"]), int(row["away_score"])
        
        if h in baseline_stats:
            b = baseline_stats[h]
            b["played"] += 1
            b["gf"] += hs
            b["ga"] += aws
            if hs > aws:
                b["won"] += 1
                b["pts"] += 3
            elif hs < aws:
                b["lost"] += 1
            else:
                b["drawn"] += 1
                b["pts"] += 1

        if a in baseline_stats:
            b = baseline_stats[a]
            b["played"] += 1
            b["gf"] += aws
            b["ga"] += hs
            if aws > hs:
                b["won"] += 1
                b["pts"] += 3
            elif aws < hs:
                b["lost"] += 1
            else:
                b["drawn"] += 1
                b["pts"] += 1

    # 2. Get predictions for all upcoming matches (pre-calculate to speed up simulation loops)
    match_probs = []
    for _, row in upcoming_matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        # Use predict_fast to get probabilities
        pred = predictor.predict_fast(h, a, stage="group", is_home=0) # neutral venue in World Cup
        probs = pred["probabilities"]
        match_probs.append({
            "home": h,
            "away": a,
            "p_win": probs["win"],   # Home win
            "p_draw": probs["draw"], # Draw
            "p_loss": probs["loss"]  # Away win (Home loss)
        })

    # 3. Simulation loop
    # We track rank counts for each team: {team: [rank_1_count, rank_2_count, rank_3_count, rank_4_count]}
    rank_counts = {team: [0, 0, 0, 0] for team in teams}
    points_accumulator = {team: [] for team in teams}
    gd_accumulator = {team: [] for team in teams}

    for _ in range(n_simulations):
        # Start with a copy of baseline stats
        sim_stats = {t: dict(baseline_stats[t]) for t in teams}

        # Simulate each upcoming match
        for m in match_probs:
            h, a = m["home"], m["away"]
            # Draw outcome based on probabilities
            r = random.random()
            if r < m["p_win"]:
                outcome = "Win"
            elif r < m["p_win"] + m["p_draw"]:
                outcome = "Draw"
            else:
                outcome = "Loss"

            hg, ag = sample_match_score(outcome)

            if h in sim_stats:
                s = sim_stats[h]
                s["played"] += 1
                s["gf"] += hg
                s["ga"] += ag
                if outcome == "Win":
                    s["won"] += 1
                    s["pts"] += 3
                elif outcome == "Draw":
                    s["drawn"] += 1
                    s["pts"] += 1
                else:
                    s["lost"] += 1

            if a in sim_stats:
                s = sim_stats[a]
                s["played"] += 1
                s["gf"] += ag
                s["ga"] += hg
                if outcome == "Loss":
                    s["won"] += 1
                    s["pts"] += 3
                elif outcome == "Draw":
                    s["drawn"] += 1
                    s["pts"] += 1
                else:
                    s["lost"] += 1

        # Calculate final group standings for this run
        run_ranking = []
        for team in teams:
            s = sim_stats[team]
            run_ranking.append({
                "team": team,
                "pts": s["pts"],
                "gd": s["gf"] - s["ga"],
                "gf": s["gf"]
            })
        
        # Sort by Pts desc, GD desc, GF desc, random tiebreaker (or alphabetical)
        # Note: in real tournament it might go to fair play points or drawing of lots, so we sort alphabetically as stable fallback
        run_ranking = sorted(
            run_ranking,
            key=lambda x: (x["pts"], x["gd"], x["gf"], x["team"]),
            reverse=True
        )

        # Update accumulators
        for rank_idx, r_item in enumerate(run_ranking[:4]): # Keep it to top 4 (or size of group)
            team = r_item["team"]
            if rank_idx < 4:
                rank_counts[team][rank_idx] += 1
            points_accumulator[team].append(r_item["pts"])
            gd_accumulator[team].append(r_item["gd"])

    # 4. Aggregate results
    results = {}
    for team in teams:
        avg_pts = np.mean(points_accumulator[team])
        avg_gd = np.mean(gd_accumulator[team])
        ranks_prob = [count / n_simulations for count in rank_counts[team]]
        results[team] = {
            "team": team,
            "avg_pts": avg_pts,
            "avg_gd": avg_gd,
            "prob_1st": ranks_prob[0],
            "prob_2nd": ranks_prob[1],
            "prob_3rd": ranks_prob[2],
            "prob_4th": ranks_prob[3],
            "prob_qualify": ranks_prob[0] + ranks_prob[1] # Top 2 qualify
        }

    return results
