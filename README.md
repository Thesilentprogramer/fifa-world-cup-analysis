# FIFA World Cup Match Predictor

A machine learning system that predicts international football match outcomes (Win / Draw / Loss) using martj42 international results enriched with StatsBomb event data, Elo ratings, and player aggregates.

## Features

- **Match Outcome Predictor** — calibrated probability estimates for any two national teams
- **SHAP explanations** — understand why the model favors one side
- **Temporal validation** — trained on men's internationals before 2018, validated on 2018 WC, tested on 2022 WC
- **Elo + form + H2H** — 28,000+ training rows from 21k+ matches since 2000

## Quick Start

```bash
# 1. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (martj42 + StatsBomb + odds)
python scripts/download_data.py
# Or international only (faster):
python scripts/download_data.py --international-only

# 4. Build features
python scripts/build_features.py

# 5. Train model
python scripts/train_model.py

# 6. Run app
streamlit run app/main.py
```

## Project Structure

```
fifa-world-cup-analysis/
├── data/raw/           # martj42, StatsBomb, odds, FBref cache
├── data/processed/     # Engineered feature tables
├── models/             # Trained model artifacts
├── src/                # Core Python modules
├── scripts/            # Data download, feature build, training
├── app/                # Streamlit application
└── notebooks/          # EDA and model development
```

## Data Sources

| Source | License | Content |
|--------|---------|---------|
| [martj42/international_results](https://github.com/martj42/international_results) | Free | ~49k men's international match results |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Free | Match events, lineups, xG (enrichment) |
| [football-data.co.uk](https://www.football-data.co.uk/) | Free | World Cup bookmaker odds (when available) |
| FBref via soccerdata | Free | Player aggregates (optional; goalscorers fallback) |

## Model Evaluation

Honest held-out evaluation on men's FIFA World Cup matches only. Three-class accuracy on full Win/Draw/Loss is inherently capped (~55–65% is strong); 85–89% is not a realistic target for this task.

| Split | Rows | Log-loss | Brier | Accuracy |
|-------|------|----------|-------|----------|
| Validation (2018 WC) | 128 | 0.80 | 0.16 | **61.7%** |
| Test (2022 WC, held-out) | 128 | 3.56 | 0.19 | **60.2%** |
| High-confidence test (≥55%) | 88 | — | — | **61.4%** |

Training pool: **28,672** perspective rows before 2018-06-01 (21,195 matches since 2000). Features include Elo ratings, rolling form, tournament-specific form, rest days, H2H history, StatsBomb xG/shots (where available), and FBref/goalscorer aggregates. Probabilities calibrated via isotonic regression on 2018 World Cup matches.

## Deploy to Streamlit Community Cloud

1. Push this repo to a public GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io) and create a new app
3. Set main file path to `app/main.py`
4. Deploy — models and inference data are bundled in the repo

## License

MIT
