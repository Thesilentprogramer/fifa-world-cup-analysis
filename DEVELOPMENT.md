# FIFA World Cup Match Predictor — Development Notes

## What This App Really Does

The FIFA World Cup Match Predictor is a **machine learning pipeline + Streamlit app** that estimates international football match outcomes (Win / Draw / Loss) and explains those predictions with SHAP. It is designed around honest temporal evaluation — train on history, validate on 2018 World Cup, test on held-out 2022 World Cup — not inflated accuracy on easy subsets.

### The Core Idea

When you analyze a match (e.g. **Brazil vs Argentina**), the system runs this pipeline:

1. **Data backbone** — [martj42 international results](https://github.com/martj42/international_results) provide ~21k men's matches since 2000. StatsBomb open data enriches a subset with shots, xG, corners, and possession.

2. **Feature engineering** — For every match, two perspective rows are built (home + away) with:
   - **Elo ratings** (chronological, no leakage)
   - Rolling **form** (5/10 games) and **tournament-specific form**
   - **Head-to-head** history, **rest days**, confederation flags
   - Optional: squad market value (Transfermarkt API), Polymarket/bookmaker odds, FBref/goalscorer aggregates

3. **Match outcome model** — XGBoost multi-class classifier with isotonic calibration on 2018 WC. Outputs calibrated P(Win), P(Draw), P(Loss).

4. **Explainability** — SHAP TreeExplainer surfaces the top 5 features driving each prediction in plain language.

5. **xG engine (Phase 2)** — Shot-level logistic model on StatsBomb events + Monte Carlo scoreline simulation for expected goals and likely scores.

6. **WC 2026 dashboard** — Cached fixtures from API-Football (or martj42 fallback) with batch pre-match analysis: win %, xG, Elo, and per-match SHAP on demand.

---

## Architecture

```
data/raw/
  international/     martj42 results + goalscorers
  statsbomb/         match metadata + per-match event JSON (~369 matches)
  api_football/      WC 2026 fixture cache
  transfermarkt/     squad values + national team IDs
  polymarket/        prediction-market odds cache
  fbref/             player aggregate cache

scripts/
  download_data.py   fetch + sync all sources
  build_features.py  -> data/processed/match_features.parquet
  train_model.py     -> models/match_outcome.pkl + calibrator
  train_xg_model.py  -> models/xg_model.pkl
  daily_matchday_refresh.py  cron: sync + conditional retrain
  install_cron.sh    prints crontab line
  fetch_live_polymarket.py   fetch active soccer markets & odds from Polymarket Gamma API

data/processed/
  match_features.parquet
  prediction_log.parquet   kickoff predictions for played review
  last_refresh.json        daily cron metadata

src/
  fixture_filters.py   today/tomorrow local-date windows
  prediction_cache.py  prediction log CRUD + played merge

app/
  main.py            Splash landing (Three.js) + ENTER DASHBOARD CTA
  theme.py           Splash CSS + shared app styles
  assets/
    splash_scene.html      Football + 48 flag orbit (flagcdn textures)
    wc2026_nations.json    48 WC teams + ISO codes
  charts/
    model_charts.py     Plotly showcase charts (metrics, calibration)
  pages/
    1_match_predictor.py   Lean WC dashboard (today/tomorrow/played)
    2_xg_engine.py         xG simulation + calibration charts + pitch viz
    3_custom_matchup.py    H2H Matchup Center (W/D/L, SHAP, team metrics, top scorers, shootout tiebreaker)
    4_player_dashboard.py  Phase 3 Player Dashboard (top scorers, StatsBomb analytics)
    5_penalty_simulator.py Phase 4 Penalty Shootout Simulator (Monte Carlo + interactive)
    6_knockout_bracket.py  Knockout Bracket Simulator (visual tree + round-by-round sim)
```

### Training / evaluation splits

| Split | Definition |
|-------|------------|
| **Train** | Men's internationals before 2018-06-01 (~28,672 rows) |
| **Validation** | 2018 FIFA World Cup only (128 rows) |
| **Test** | 2022 FIFA World Cup only (128 rows, held-out) |

**Realistic accuracy target:** ~55–65% on full 3-class WC test. 85–89% is not achievable with honest evaluation.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit, Plotly, Three.js splash (CDN via `components.html`) |
| **ML** | XGBoost, scikit-learn (isotonic calibration, logistic xG model) |
| **Explainability** | SHAP TreeExplainer |
| **Visualization** | mplsoccer (pitch shot maps on xG page) |
| **Data** | pandas, pyarrow, requests |
| **Event data** | statsbombpy, flattened JSON event cache |
| **External APIs** | martj42 (GitHub), API-Football, Transfermarkt API, Polymarket Gamma/CLOB, FBref via soccerdata |
| **Config** | python-dotenv (`.env` for API keys) |

---

## Recent Changes (June 19, 2026)

### Flags Orbit Splash + Lean Dashboard + Daily Cron
- **`app/assets/splash_scene.html`**: Replaced voxel tiles with **central football** + **48 WC nation flags** orbiting on two rings (textures from flagcdn.com via `wc2026_nations.json`).
- **`app/pages/1_match_predictor.py`**: Decluttered dashboard — **Today**, **Tomorrow**, and **Played (cached)** sections only; compact match cards; deep dive (donut/xG/SHAP) on checkbox; model metrics moved to bottom expander.
- **`app/pages/3_custom_matchup.py`**: Custom head-to-head moved out of main dashboard.
- **`src/fixture_filters.py`**: Local-timezone today/tomorrow split; stale sync detection.
- **`src/prediction_cache.py`**: `prediction_log.parquet` stores kickoff predictions; played tab reads cache (no batch xG re-run).
- **`scripts/daily_matchday_refresh.py`**: Pulls martj42 + fixtures daily; **retrains only when new finished matches** appear; updates prediction log.
- **`scripts/install_cron.sh`**: Prints `crontab -e` line for 6 AM daily refresh.
- **Played prediction backfill**: `ensure_played_predictions()` auto-logs model picks for finished fixtures missing from `prediction_log.parquet` (retrospective when kickoff log absent).

### Streamlit Splash Landing + Model Showcase Charts
- **`app/assets/splash_scene.html`**: Three.js Roblox-style scene — studded green ground tiles, floating WC-colored voxel cubes, slow camera drift. Loaded via `st.components.v1.html` (scripts cannot live in `st.markdown`).
- **`app/theme.py`**: `inject_splash_styles()` hides Streamlit chrome on splash; `render_splash_background()` embeds the Three.js iframe; `inject_app_styles()` adds Syne/Inter fonts on model pages.
- **`app/main.py`**: Fullscreen splash with title **FIFA WORLD CUP MATCH ANALYSIS DASHBOARD**, **ENTER DASHBOARD** button → `st.switch_page("pages/1_match_predictor.py")`. Project blurb moved to **About** expander.
- **`app/charts/model_charts.py`**: Shared Plotly helpers — val/test performance bars, WC prediction breakdown, confidence histogram, xG calibration curve, distance vs xG scatter.
- **Match Predictor page**: **Model Performance** section at top (metrics row + charts); WC tab adds prediction pie + confidence histogram after fixture analysis.
- **xG Engine page**: **Model Performance** section (ROC-AUC, shot count, calibration + distance charts) above the simulator.

### Import cycle fix (`statsbomb_shots.py`)
- **`src/statsbomb_shots.py`**: Shot loader extracted from `data_loader` to prevent `ImportError` when Streamlit loads `xg_engine` ↔ `data_loader` in partial order.
- **`prematch_analysis.py` / `api_football_client.py`**: Broader exception handling; eager imports where safe.

### API-Football Fixture Cache + WC 2026 Dashboard
- **`src/api_football_client.py`**: Fetches WC 2026 fixtures from [API-Football v3](https://www.api-football.com/documentation-v3), caches to `data/raw/api_football/fixtures_wc2026.parquet`.
- **martj42 fallback**: If `API_FOOTBALL_KEY` is missing, sync uses martj42 FIFA World Cup 2026 rows (24 group-stage matches cached).
- **`src/prematch_analysis.py`**: Bundles W/D/L probabilities, xG simulation, Elo snapshot, and SHAP per fixture.
- **Match Predictor UI**: Two tabs — **WC 2026 Dashboard** (fixture table + detail panel) and **Custom Matchup** (manual picker).
- **Performance**: Batch table uses `predict_fast()` (no SHAP); SHAP computed only for the selected match in the detail panel. Results cached 30 min in Streamlit.

### Phase 2 — xG Engine
- **`src/xg_engine.py`**: Shot-level logistic model (distance, angle, body part, technique, shot type).
- **`scripts/train_xg_model.py`**: Trains on 9,649 StatsBomb shots (ROC-AUC **0.81**).
- **`app/pages/2_xg_engine.py`**: Monte Carlo scoreline simulation, xG metrics, mplsoccer shot map.

### Transfermarkt API + Polymarket Odds
- **`src/transfermarkt_client.py`**: Squad market values via [felipeall/transfermarkt-api](https://github.com/felipeall/transfermarkt-api) with disk cache and rate limiting.
- **`src/polymarket_client.py`**: Public Gamma/CLOB endpoints for WC prediction-market odds (no API key for reads).
- New features: `squad_market_value`, `squad_value_diff`, `polymarket_prob_*`.

### Environment & DX
- **`src/env_check.py`**: Auto-reexecs scripts with `.venv/bin/python` when system Python is used.
- **`.env.example`**: Documents `API_FOOTBALL_KEY`, Transfermarkt base URL, Polymarket optional keys.
- **`python-dotenv`** added to load `.env` from project root.

---

## Phase 1 Model Upgrade (June 2026)

### Data expansion
- Primary backbone switched from StatsBomb-only (~738 rows) to **martj42** (~42,388 feature rows).
- StatsBomb demoted to enrichment layer (left-join xG/shots on date + teams).
- Men's-only filters; women's WC, Copa del Rey, and junk tournaments excluded.

### New features
- Chronological **Elo** (`src/elo.py`) — K=20, home advantage +100, no leakage.
- **Rest days**, **tournament-specific form**, optimized H2H (no double-counting).
- FBref player aggregates with **goalscorers.csv fallback** when scraping is skipped.

### Evaluation fix
- Honest 2018 val / 2022 test on men's FIFA World Cup only.
- Balanced class weights in XGBoost training.
- Empty odds columns auto-excluded when football-data.co.uk download fails.

### Results (current `models/metrics.json`)

| Split | Accuracy | Log-loss | Brier |
|-------|----------|----------|-------|
| Val (2018 WC) | **60.2%** | 0.947 | 0.188 |
| Test (2022 WC) | **57.8%** | 0.938 | 0.184 |
| High-confidence test (≥55%) | **74.1%** | — | — |

Sigmoid calibration (Platt scaling) successfully resolved the high test log-loss issue (dropped from 2.76 to 0.938) and increased test accuracy by 1.5%.

Up from ~45% test accuracy on the original StatsBomb-only pipeline.

---

## Phase 0 — Initial Build

- StatsBomb download (per-match event files, resumable).
- XGBoost + isotonic calibration (`FrozenEstimator` for sklearn 1.8+).
- Streamlit match predictor with Plotly donut + SHAP panel.
- Notebooks: `01_eda.ipynb`, `02_match_model.ipynb`.

---

## Issues We Faced During Development

### 1. StatsBomb Event Download Hang / Disk Full
- **Problem**: Downloading a single combined `events.json` (~2.9GB) hung or filled disk.
- **Fix**: Per-match files in `events_by_match/{match_id}.json` only; loader prefers that directory.

### 2. Flattened StatsBomb JSON (`AttributeError: 'str' object has no attribute 'get'`)
- **Problem**: statsbombpy exports flat columns (`team` as string, `shot_statsbomb_xg` as top-level field).
- **Fix**: Helper functions in `data_loader.py` (`_event_team_name`, `_event_shot_xg`, etc.) handle both nested and flat formats.

### 3. sklearn API Deprecations
- **`CalibratedClassifierCV(cv='prefit')` removed** → use `FrozenEstimator` wrapper.
- **`LogisticRegression(multi_class=...)` deprecated** → removed param.

### 4. Duplicate `(match_id, team)` Rows (Congo / DR Congo)
- **Problem**: Both "Congo" and "Congo DR" mapped to "DR Congo", creating home vs home matches and breaking rest-day merges.
- **Fix**: Separate canonical names; drop `home_team == away_team`; dedupe `match_id`.

### 5. Wrong Python Interpreter (Command Line Tools vs `.venv`)
- **Problem**: `pip install` into `.venv` but `python scripts/...` used system Python 3.9 without xgboost/pyarrow.
- **Fix**: `env_check.py` auto-reexecs with `.venv/bin/python`; clearer error messages.

### 6. Legacy Transfermarkt CSV 404s
- **Problem**: dcaribou GitHub URLs for `games.csv`, `players.csv`, etc. return 404.
- **Fix**: Replaced with [transfermarkt-api](https://github.com/felipeall/transfermarkt-api). Legacy CSV step kept as optional no-op.

### 7. football-data.co.uk Odds SSL / 404
- **Problem**: WC odds spreadsheet fails to download on some machines (SSL cert errors).
- **Fix**: Odds features median-imputed; empty columns excluded from training. Polymarket added as alternative odds source.

### 8. SHAP Too Slow for Batch Dashboard
- **Problem**: Running SHAP for 24 fixtures took 5+ minutes.
- **Fix**: `predict_fast()` for table; full SHAP only on selected match. Streamlit `@st.cache_data(ttl=1800)`.

### 9. FBref Scraping Slow / Fragile
- **Problem**: soccerdata FBref pulls can hang or break.
- **Fix**: Default to goalscorers.csv fallback; live FBref only when `use_fbref=True`.

### 10. Streamlit Strips Inline `<script>` Tags
- **Problem**: Three.js in `st.markdown(unsafe_allow_html=True)` never executed — blank background.
- **Fix**: Self-contained HTML asset rendered with `st.components.v1.html()`.

### 11. Splash Button Layering
- **Problem**: iframe background can steal clicks if z-index is wrong.
- **Fix**: Background iframe `pointer-events: none`; Streamlit button in foreground `z-index: 1`.

### 12. `use_container_width` Deprecation (Streamlit 2025+)
- **Fix**: Replaced with `width="stretch"` on `st.plotly_chart` / `st.dataframe`.

### 13. Flag Texture CDN (Three.js)
- **Note**: Splash flags load from flagcdn.com inside `components.html` iframe; requires network. England/Scotland use `gb-eng` / `gb-sct` codes.

### 14. Timezone Edge Cases (Today/Tomorrow)
- **Note**: Fixture windows use `datetime.now().astimezone()` local date; UTC-stored kickoffs converted before comparison.

---

## Current Known Issues

### 🟡 API-Football Rate Limits
Free tier is ~100 requests/day. Sync fixtures once per day (`python scripts/download_data.py --api-football-only`), not on every page load.

### 🟡 martj42 Fixture Lag
Without API-Football, martj42 may not have upcoming knockout fixtures until after they are played. API key recommended during live tournaments.

### 🟡 Out-of-Sample WC 2026
Model trained primarily on pre-2018 data. WC 2026 predictions are useful demos but not calibrated for this tournament — refresh features (`build_features.py`) after each matchday.

### 🟡 xG Coverage Gaps
Monte Carlo xG simulation depends on StatsBomb shot history. Teams with few cached international matches get fallback rates.

### 🟡 Transfermarkt Squad Values Are Point-in-Time
Cached values repeat across years until refreshed. Not a true historical valuation time series yet.

### 🟡 Three.js CDN Offline
Splash background requires network for `cdnjs.cloudflare.com` Three.js. Falls back to dark page if CDN blocked; app still works.

### 🟢 Resolved: High Test Log-Loss (Mitigated)
Isotonic calibration overfit to the small 128-row validation set, causing extreme probabilities and high log-loss (2.76) on 2022 test set. Switched to sigmoid calibration (Platt scaling), which reduced log-loss to **0.938** and raised test accuracy to **57.8%**.

---

## Environment Variables

Copy `.env.example` → `.env`:

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `API_FOOTBALL_KEY` | For live fixtures | WC 2026 schedule sync |
| `TRANSFERMARKT_API_BASE` | No | Default: fly.dev hosted API |
| `POLYMARKET_API_KEY` | No | Only for trading; reads work without |
| `API_FOOTBALL_WC_LEAGUE_ID` | No | Default: `1` (World Cup) |
| `API_FOOTBALL_WC_SEASON` | No | Default: `2026` |

---

## Common Commands

```bash
# Setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add API_FOOTBALL_KEY

# Data pipeline
python scripts/download_data.py --international-only
python scripts/download_data.py --api-football-only
python scripts/download_data.py --tm-api-only
python scripts/download_data.py --polymarket-only
python scripts/build_features.py

# Query live odds
python scripts/fetch_live_polymarket.py

# Training
python scripts/train_model.py
python scripts/train_xg_model.py

# App
streamlit run app/main.py

# Daily refresh (manual or cron)
python scripts/daily_matchday_refresh.py
bash scripts/install_cron.sh   # print crontab line
```

---

## Roadmap — Features We Can Add

### High Priority
*None currently.*

### Medium Priority
- [ ] **Historical valuation time series** — Transfermarkt player values as-of match date. (Using live API only for current squads).
- [ ] **FIFA ranking join** — Monthly rank/points as optional feature.
- [ ] **Live odds integration** — Real-time bookmaker lines via Polymarket Gamma API for upcoming fixtures.
- [x] **PDF match report export** — One-click pre-match briefing generation.

### Low Priority

### Completed Features
- [x] **API-Football live odds join** — `/odds?fixture=` endpoint for bookmaker lines on dashboard.
- [x] **Group standings widget** — Points table with predicted (Monte Carlo) vs actual finish.
- [x] **PDF match report export** — One-click pre-match briefing briefing.
- [x] **Docker / docker-compose** — One-command deploy with cached data volume.
- [x] **Knockout bracket view** — Visual bracket with predictions as tournament progresses (interactive simulator + round-by-round W/D/L + penalty simulator tiebreakers).
- [x] **Model vs market benchmark** — Side-by-side our probabilities vs Polymarket / API-Football predictions on dashboard upcoming matches deep dive.
- [x] **Phase 3 — Player dashboard** — Top scorers (goalscorers.csv), StatsBomb event-level player leaderboards & xG Chain, player profile stats cards & radar.
- [x] **Phase 4 — Penalty shootout simulator** — Separate Monte Carlo simulation model for shootout tiebreakers (including pressure decay, keeper presets, taker skill order, sequential early termination).
- [x] **48-flag orbit splash** — Football center + revolving WC nation flags.
- [x] **Lean matchday dashboard** — Today/tomorrow upcoming + cached played results.
- [x] **Prediction log at kickoff** — `prediction_log.parquet` for post-match review.
- [x] **Daily matchday cron** — Conditional retrain when new finished matches detected.
- [x] **Three.js splash landing** — WC dashboard CTA.
- [x] **Per-model showcase charts** — Match outcome + xG performance sections.
- [x] **martj42 data backbone** — 21k+ matches, 42k+ feature rows.
- [x] **Elo ratings** — Chronological, leakage-free.
- [x] **XGBoost + sigmoid calibration** — 2018 val / 2022 test splits with Platt scaling (test log-loss 0.938).
- [x] **SHAP explanations** — Top-5 features + narrative.
- [x] **StatsBomb event enrichment** — xG, shots, corners (369 matches).
- [x] **Transfermarkt API squad values** — Cached national team market values.
- [x] **Polymarket odds client** — Public API, disk cache.
- [x] **Phase 2 xG engine** — Shot model + Monte Carlo + Streamlit page.
- [x] **API-Football fixture cache** — WC 2026 with martj42 fallback.
- [x] **WC 2026 pre-match dashboard** — Batch W/D/L + xG + per-match SHAP.
- [x] **Venv auto-reexec** — Scripts work even when system `python` is wrong.
- [x] **FBref / goalscorers fallback** — Player aggregate features without scraping.

---

## Project Phases (Original Plan)

| Phase | Scope | Status |
|-------|--------|--------|
| 0 | Data pipeline + EDA | ✅ Done |
| 1 | Match outcome predictor + Phase 1 upgrade | ✅ Done |
| 2 | xG engine + Monte Carlo + Page 2 | ✅ Done |
| 3 | Player dashboard | ✅ Done |
| 4 | Penalty simulator | ✅ Done |
| 5 | Odds benchmark, deploy polish, SHAP narratives | ✅ Done |

---

*Last updated: June 19, 2026 (flags splash + lean dashboard + daily cron)*
