# FIFA World Cup 2026 — Engineered Match Dataset

**Companion dataset for the notebook:**  
[FIFA World Cup 2026 — End-to-End ML Analysis](https://www.kaggle.com/code/shubhamindulkar/fifa-world-cup-2026-end-to-end-ml-analysis)

---

## What's in this dataset?

This dataset contains pre-engineered features, model metadata, and squad data built from multiple public football data sources. It is designed to be used directly in ML training pipelines — no raw data wrangling required.

| File | Description |
|------|-------------|
| `match_features.csv` | **Main feature table** — 28,000+ team-perspective rows for men's international matches (2000–2025). Each row is one team's view of one match. |
| `squad_values_by_year.csv` | Aggregated squad market values per national team per year, sourced from [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets). |
| `national_team_ids.json` | Transfermarkt internal IDs for national teams (used for squad value joins). |
| `feature_columns.json` | Ordered list of feature column names used during XGBoost model training. |
| `team_encoder.json` | Sorted list of all teams seen during training (for encoding/lookup). |
| `metrics.json` | Validation and test set metrics from the trained XGBoost model. |

---

## `match_features.csv` — column reference

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | str | Unique match identifier |
| `match_date` | datetime | Date of the match |
| `team` / `opponent` | str | Canonical team names (normalized) |
| `goals_for` / `goals_against` | int | Full-time goals |
| `outcome` | int | 0 = Loss, 1 = Draw, 2 = Win (from team's perspective) |
| `is_home` | int | 1 if playing at home venue, 0 otherwise |
| `is_neutral` | int | 1 if neutral venue |
| `is_knockout` | int | 1 if knockout stage match |
| `team_elo` / `opponent_elo` | float | Pre-match ELO ratings (K=20, home adv=100) |
| `elo_diff` | float | `team_elo - opponent_elo` |
| `form_5` / `form_10` | float | Rolling points sum over last 5/10 matches |
| `form_diff_5` / `form_diff_10` | float | Form advantage vs opponent |
| `goals_for_avg_5/10` / `goals_against_avg_5/10` | float | Rolling goal averages |
| `tournament_form_5` | float | Rolling form within same tournament type |
| `tournament_gf_avg_5` / `tournament_ga_avg_5` | float | Rolling goals within tournament type |
| `team_rest_days` / `rest_days_diff` | float | Days since last match; rest advantage |
| `h2h_n` / `h2h_wins` / `h2h_draws` / `h2h_losses` | int | Head-to-head history (last 10 meetings) |
| `h2h_gf` / `h2h_ga` | float | H2H goals scored/conceded |
| `same_confederation` | int | 1 if both teams from same confederation |
| `squad_market_value` | float | Squad value in € (from Transfermarkt, where available) |
| `squad_value_diff` | float | Squad value advantage vs opponent |
| `implied_prob_win/draw/loss` | float | Bookmaker-implied probabilities (where available) |

---

## Data sources

| Source | License |
|--------|---------|
| [martj42/international_results](https://github.com/martj42/international_results) | Free / Open |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Free (non-commercial) |
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | MIT |

---

## Train/Validation/Test split used

| Split | Period | Tournament | Rows |
|-------|--------|-----------|------|
| Train | 2000 – 2018-05-31 | All | ~26,000 |
| Validation | 2018 Jun–Jul | FIFA World Cup 2018 | 128 |
| Test (held-out) | 2022 Nov–Dec | FIFA World Cup 2022 | 128 |

---

## Model metrics

| Split | Accuracy | Log-loss | Brier |
|-------|----------|----------|-------|
| Validation (2018 WC) | **61.7%** | 0.80 | 0.16 |
| Test (2022 WC) | **60.2%** | 3.56 | 0.19 |
| High-confidence ≥55% | **61.4%** | — | — |

---

## Related links

- 📓 [Kaggle Notebook](https://www.kaggle.com/code/shubhamindulkar/fifa-world-cup-2026-end-to-end-ml-analysis)
- 💻 [GitHub Repository](https://github.com/Thesilentprogramer/fifa-world-cup-analysis) — full Streamlit app with live predictions, SHAP explanations, xG engine, knockout bracket
- 🌿 [kaggle-notebook branch](https://github.com/Thesilentprogramer/fifa-world-cup-analysis/tree/kaggle-notebook)
