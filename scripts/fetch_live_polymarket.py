#!/usr/bin/env python3
"""Fetch active football/soccer markets and implied probabilities from Polymarket Gamma API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env_check import ensure_project_deps

ensure_project_deps()


def fetch_active_football_markets(limit: int = 100) -> list[dict]:
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        "active": "true",
        "limit": limit,
    }
    
    print(f"Connecting to Polymarket Gamma API ({url})...")
    try:
        resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": "fifa-wc-analysis/1.0"})
        if resp.status_code != 200:
            print(f"Error: API returned status code {resp.status_code}", file=sys.stderr)
            return []
            
        data = resp.json()
        if not isinstance(data, list):
            print("Error: Unexpected JSON response structure (expected a list of markets)", file=sys.stderr)
            return []
            
        return data
    except requests.RequestException as e:
        print(f"Connection Error: {e}", file=sys.stderr)
        return []


def filter_football_markets(markets: list[dict]) -> list[dict]:
    football_keywords = {
        "soccer", "football", "world cup", "euro 20", "copa america",
        "mls", "premier league", "champions league", "laliga", "serie a", "bundesliga"
    }
    
    filtered = []
    for m in markets:
        title = (m.get("title") or m.get("question") or "").lower()
        desc = (m.get("description") or "").lower()
        category = (m.get("category") or "").lower()
        group_title = (m.get("groupTitle") or "").lower()
        
        # Check title, category, description, and group title for keywords
        is_football = any(
            kw in title or kw in desc or kw in category or kw in group_title
            for kw in football_keywords
        )
        if is_football:
            filtered.append(m)
            
    return filtered


def parse_outcome_prices(market: dict) -> dict[str, float]:
    outcomes = market.get("outcomes") or market.get("shortOutcomes") or []
    prices = market.get("outcomePrices") or []
    
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            pass
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except Exception:
            pass
            
    if not outcomes or not prices or len(outcomes) != len(prices):
        return {}
        
    return {str(o): float(p) for o, p in zip(outcomes, prices)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live soccer/football markets from Polymarket")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of active markets to fetch")
    parser.add_argument("--json", action="store_true", help="Output raw JSON data")
    args = parser.parse_args()

    raw_markets = fetch_active_football_markets(limit=args.limit)
    if not raw_markets:
        print("No active markets returned or connection failed.")
        sys.exit(1)

    football_markets = filter_football_markets(raw_markets)
    
    if args.json:
        print(json.dumps(football_markets, indent=2))
        return

    print(f"\nFound {len(football_markets)} active football/soccer markets out of {len(raw_markets)} active markets fetched:\n")
    print("=" * 80)
    
    for idx, m in enumerate(football_markets, 1):
        title = m.get("title") or m.get("question") or "N/A"
        category = m.get("category") or "N/A"
        market_id = m.get("id") or m.get("conditionId") or "N/A"
        prices = parse_outcome_prices(m)
        
        print(f"{idx}. Title: {title}")
        print(f"   Category: {category} | Market ID: {market_id}")
        if prices:
            print("   Implied Probabilities / Prices:")
            for outcome, price in prices.items():
                print(f"     - {outcome}: {price:.1%} (${price:.2f})")
        else:
            print("   Implied Probabilities: N/A (No outcome prices available)")
        print("-" * 80)


if __name__ == "__main__":
    main()
