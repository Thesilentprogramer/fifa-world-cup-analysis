"""Project configuration from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root when present
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

# Transfermarkt API — https://github.com/felipeall/transfermarkt-api
TRANSFERMARKT_API_BASE = os.getenv(
    "TRANSFERMARKT_API_BASE",
    "https://transfermarkt-api.fly.dev",
)
TRANSFERMARKT_RATE_LIMIT_SEC = float(os.getenv("TRANSFERMARKT_RATE_LIMIT_SEC", "2.5"))

# Polymarket — public read endpoints, no key required for market data
POLYMARKET_GAMMA_BASE = os.getenv("POLYMARKET_GAMMA_BASE", "https://gamma-api.polymarket.com")
POLYMARKET_CLOB_BASE = os.getenv("POLYMARKET_CLOB_BASE", "https://clob.polymarket.com")
POLYMARKET_RATE_LIMIT_SEC = float(os.getenv("POLYMARKET_RATE_LIMIT_SEC", "0.5"))
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET = os.getenv("POLYMARKET_API_SECRET", "")
POLYMARKET_API_PASSPHRASE = os.getenv("POLYMARKET_API_PASSPHRASE", "")

# API-Football — https://www.api-football.com/documentation-v3
API_FOOTBALL_BASE = os.getenv("API_FOOTBALL_BASE", "https://v3.football.api-sports.io")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_WC_LEAGUE_ID = int(os.getenv("API_FOOTBALL_WC_LEAGUE_ID", "1"))
API_FOOTBALL_WC_SEASON = int(os.getenv("API_FOOTBALL_WC_SEASON", "2026"))
API_FOOTBALL_RATE_LIMIT_SEC = float(os.getenv("API_FOOTBALL_RATE_LIMIT_SEC", "1.0"))
