"""Phase 4 — Penalty Shootout Simulator (Monte Carlo engine).

simulate_shootout(team_a_takers, team_b_takers, team_a_keeper_skill, team_b_keeper_skill,
                  n_simulations=10_000)
  -> dict with win_prob_a, win_prob_b, draw_prob (rare), kick_log (one sample run)

Taker skill: 0.0–1.0 modifier on base conversion rate (~0.76).
Keeper skill: 0.0–1.0 modifier that adds extra save probability.

Historical named keeper presets are provided so users can pick well-known
penalty specialists (e.g. Emi Martínez, Livaković, Yann Sommer, etc.).
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Base probabilities
# ---------------------------------------------------------------------------
BASE_CONVERSION = 0.757          # global historical penalty conversion rate
BASE_SAVE = 1.0 - BASE_CONVERSION  # ~24.3%
PRESSURE_DECAY = 0.03            # conversion drops this much per "must-score" round ≥6
FIRST_KICKER_ADVANTAGE = 0.012   # tiny boost for team kicking first


# ---------------------------------------------------------------------------
# Named goalkeeper presets  (save_modifier: how much they boost save %)
# ---------------------------------------------------------------------------
KEEPER_PRESETS: dict[str, dict] = {
    "Average keeper":            {"save_modifier": 0.00,  "flag": "🧤"},
    "Emi Martínez (Argentina)":  {"save_modifier": 0.12,  "flag": "🇦🇷"},
    "Dominik Livaković (Croatia)":{"save_modifier": 0.11, "flag": "🇭🇷"},
    "Yann Sommer (Switzerland)": {"save_modifier": 0.10,  "flag": "🇨🇭"},
    "Hugo Lloris (France)":      {"save_modifier": 0.07,  "flag": "🇫🇷"},
    "Manuel Neuer (Germany)":    {"save_modifier": 0.08,  "flag": "🇩🇪"},
    "Thibaut Courtois (Belgium)":{"save_modifier": 0.09,  "flag": "🇧🇪"},
    "Gianluigi Buffon (Italy)":  {"save_modifier": 0.10,  "flag": "🇮🇹"},
    "Ederson (Brazil)":          {"save_modifier": 0.06,  "flag": "🇧🇷"},
    "Jordan Pickford (England)": {"save_modifier": 0.10,  "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "Andriy Lunin (Ukraine)":    {"save_modifier": 0.09,  "flag": "🇺🇦"},
}

# ---------------------------------------------------------------------------
# Named penalty-taker skill presets
# ---------------------------------------------------------------------------
TAKER_PRESETS: dict[str, dict] = {
    "Average player":          {"skill": 0.00},
    "Expert (specialist)":     {"skill": 0.10},
    "Reliable":                {"skill": 0.05},
    "Nervous":                 {"skill": -0.08},
    "Poor record":             {"skill": -0.12},
}


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _conversion_prob(
    taker_skill: float,
    keeper_save_modifier: float,
    round_num: int,
    must_score: bool,
    first_kicker: bool,
) -> float:
    """Compute the probability a single penalty is converted."""
    conv = BASE_CONVERSION + taker_skill
    conv -= keeper_save_modifier
    # Pressure decay for sudden-death rounds
    if round_num > 5:
        pressure_rounds = round_num - 5
        conv -= PRESSURE_DECAY * pressure_rounds
    if must_score:
        conv -= 0.025  # extra pressure
    if first_kicker:
        conv += FIRST_KICKER_ADVANTAGE
    return float(np.clip(conv, 0.30, 0.98))


def simulate_one_shootout(
    team_a_takers: list[float],   # list of taker skill modifiers
    team_b_takers: list[float],
    keeper_a_save_modifier: float,  # team_a's keeper's save bonus
    keeper_b_save_modifier: float,
    rng: np.random.Generator,
    first_kicker: Literal["a", "b"] = "a",
) -> tuple[str, list[dict]]:
    """Simulate one complete shootout. Returns (winner: 'a'|'b', kick_log)."""
    score_a = 0
    score_b = 0
    kick_log: list[dict] = []
    
    kicks_a = list(team_a_takers)
    kicks_b = list(team_b_takers)

    # Determine order of teams
    t1 = "a" if first_kicker == "a" else "b"
    t2 = "b" if first_kicker == "a" else "a"
    
    MAX_ROUNDS = 30
    for r in range(1, MAX_ROUNDS + 1):
        # 1. First team kick
        t1_skill = kicks_a[(r - 1) % len(kicks_a)] if t1 == "a" else kicks_b[(r - 1) % len(kicks_b)]
        t1_keeper_save = keeper_b_save_modifier if t1 == "a" else keeper_a_save_modifier
        
        prob1 = _conversion_prob(t1_skill, t1_keeper_save, r, must_score=False, first_kicker=True)
        scored1 = bool(rng.random() < prob1)
        if t1 == "a":
            if scored1: score_a += 1
        else:
            if scored1: score_b += 1
            
        kick_log.append({
            "round": r,
            "team": t1,
            "scored": scored1,
            "score_a": score_a,
            "score_b": score_b,
            "prob": round(prob1, 3),
        })
        
        # Check early termination after first team's kick (only in first 5 rounds)
        if r <= 5:
            rem1 = 5 - r        # remaining kicks for first team in initial 5
            rem2 = 5 - (r - 1)  # remaining kicks for second team in initial 5
            
            s1 = score_a if t1 == "a" else score_b
            s2 = score_b if t1 == "a" else score_a
            
            if s1 > s2 + rem2:
                return t1, kick_log
            if s2 > s1 + rem1:
                return t2, kick_log
        
        # 2. Second team kick
        t2_skill = kicks_a[(r - 1) % len(kicks_a)] if t2 == "a" else kicks_b[(r - 1) % len(kicks_b)]
        t2_keeper_save = keeper_b_save_modifier if t2 == "a" else keeper_a_save_modifier
        
        prob2 = _conversion_prob(t2_skill, t2_keeper_save, r, must_score=False, first_kicker=False)
        scored2 = bool(rng.random() < prob2)
        if t2 == "a":
            if scored2: score_a += 1
        else:
            if scored2: score_b += 1
            
        kick_log.append({
            "round": r,
            "team": t2,
            "scored": scored2,
            "score_a": score_a,
            "score_b": score_b,
            "prob": round(prob2, 3),
        })
        
        # Check early termination after second team's kick
        if r <= 5:
            rem = 5 - r
            if score_a > score_b + rem:
                return "a", kick_log
            if score_b > score_a + rem:
                return "b", kick_log
        else:
            # Sudden death (r > 5): if scores are different after both have kicked, we have a winner
            if score_a != score_b:
                winner = "a" if score_a > score_b else "b"
                return winner, kick_log

    winner = "a" if score_a > score_b else ("b" if score_b > score_a else "a")
    return winner, kick_log


def simulate_shootout(
    team_a_takers: list[float],
    team_b_takers: list[float],
    keeper_a_save_modifier: float = 0.0,
    keeper_b_save_modifier: float = 0.0,
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> dict:
    """Run n_simulations shootouts and return aggregate stats + one sample log.

    Returns:
        win_prob_a: float (0–1)
        win_prob_b: float (0–1)
        sample_log: list[dict] — one sample kick-by-kick trace
        avg_rounds: float — average rounds per shootout
    """
    rng = np.random.default_rng(seed)
    wins_a = 0
    wins_b = 0
    total_rounds = 0
    sample_log: list[dict] = []

    for i in range(n_simulations):
        first = "a" if i % 2 == 0 else "b"  # alternate first kicker each sim
        winner, log = simulate_one_shootout(
            team_a_takers, team_b_takers,
            keeper_a_save_modifier, keeper_b_save_modifier,
            rng, first_kicker=first,
        )
        if winner == "a":
            wins_a += 1
        else:
            wins_b += 1
        total_rounds += max((k["round"] for k in log), default=5)
        if i == 0:
            sample_log = log

    return {
        "win_prob_a": wins_a / n_simulations,
        "win_prob_b": wins_b / n_simulations,
        "avg_rounds": total_rounds / n_simulations,
        "sample_log": sample_log,
        "n_simulations": n_simulations,
    }
