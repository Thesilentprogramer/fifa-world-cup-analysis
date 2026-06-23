#!/usr/bin/env python3
"""Download StatsBomb and Transfermarkt data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    INTERNATIONAL_DIR,
    MARTJ42_URLS,
    ODDS_DIR,
    STATSBOMB_DIR,
    TRANSFERMARKT_DIR,
    TRANSFERMARKT_FILES,
)

STATSBOMB_COMPETITIONS = [
    (43, "FIFA World Cup"),
    (55, "UEFA Euro"),
    (72, "Women's World Cup"),
    (87, "Africa Cup of Nations"),
    (116, "Copa America"),
]

ODDS_URLS = [
    "https://www.football-data.co.uk/worldcup.xlsx",
    "https://www.football-data.co.uk/mmz4281/WorldCup.csv",
]


def download_international(force: bool = False) -> None:
    """Download martj42 international results."""
    INTERNATIONAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MARTJ42_URLS.items():
        dest = INTERNATIONAL_DIR / f"{name}.csv"
        if dest.exists() and not force:
            print(f"  [skip] {name}.csv")
            continue
        print(f"  Downloading {name}.csv...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  Saved {dest} ({len(resp.content) // 1024} KB)")


def download_odds(force: bool = False) -> None:
    """Download World Cup odds from football-data.co.uk."""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ODDS_DIR / "world_cup_football_data.xlsx"
    if dest.exists() and not force:
        print(f"  [skip] odds file")
        return

    for url in ODDS_URLS:
        print(f"  Trying {url}...")
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                continue
            if url.endswith(".xlsx") or resp.headers.get("content-type", "").startswith("application"):
                dest.write_bytes(resp.content)
                print(f"  Saved {dest}")
                # Also export CSV for easier loading
                try:
                    df = pd.read_excel(dest)
                    csv_dest = ODDS_DIR / "world_cup_football_data.csv"
                    df.to_csv(csv_dest, index=False)
                    print(f"  Exported {csv_dest}")
                except Exception as e:
                    print(f"  Warning: could not convert xlsx to csv: {e}")
                return
            csv_dest = ODDS_DIR / "world_cup_football_data.csv"
            csv_dest.write_bytes(resp.content)
            print(f"  Saved {csv_dest}")
            return
        except Exception as e:
            print(f"  Warning: {e}")

    print("  [skip] Could not download odds — model will median-impute odds features")


def download_transfermarkt_api(teams: list[str] | None = None, force: bool = False) -> None:
    """Fetch squad market values via felipeall/transfermarkt-api."""
    from src.transfermarkt_client import SQUAD_VALUES_PATH, build_squad_values_table, load_squad_values

    if SQUAD_VALUES_PATH.exists() and not force and teams is None:
        print(f"  [skip] squad values cache ({len(load_squad_values())} rows)")
        return

    if teams is None:
        from src.data_loader import load_international_results
        matches = load_international_results()
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

    print(f"  Fetching Transfermarkt squad values for {len(teams)} teams...")
    df = build_squad_values_table(teams)
    print(f"  Cached {len(df)} squad value rows -> {SQUAD_VALUES_PATH}")


def download_polymarket_odds(force: bool = False) -> None:
    """Fetch Polymarket odds for World Cup matches (public API, no key)."""
    from src.polymarket_client import build_polymarket_odds_table, ODDS_CACHE_PATH
    from src.data_loader import load_international_results

    if ODDS_CACHE_PATH.exists() and not force:
        print(f"  [skip] Polymarket odds cache")
        return

    matches = load_international_results()
    print("  Fetching Polymarket odds for World Cup matches...")
    df = build_polymarket_odds_table(matches, force=force)
    print(f"  Cached {len(df)} Polymarket odds rows")


def download_api_football_fixtures(force: bool = False) -> None:
    """Sync FIFA World Cup 2026 fixtures from API-Football (or martj42 fallback)."""
    from src.api_football_client import FIXTURES_PATH, get_fixture_sync_meta, sync_wc_fixtures
    from src.match_predictor import get_predictor

    try:
        teams = get_predictor().teams
    except Exception:
        teams = None

    df = sync_wc_fixtures(force=force, known_teams=teams)
    meta = get_fixture_sync_meta()
    print(f"  Cached {len(df)} fixtures -> {FIXTURES_PATH}")
    if meta:
        print(f"  Source: {meta.get('source', 'unknown')}, synced: {meta.get('synced_at', '?')}")


def download_transfermarkt(force: bool = False) -> None:
    """Download Transfermarkt CSV files from GitHub."""
    TRANSFERMARKT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in TRANSFERMARKT_FILES.items():
        dest = TRANSFERMARKT_DIR / f"{name}.csv"
        if dest.exists() and not force:
            print(f"  [skip] {name}.csv already exists")
            continue
        print(f"  Downloading {name}.csv...")
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            print(f"  [skip] {name}.csv not available at URL")
            continue
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  Saved {dest}")


def download_statsbomb_matches(force: bool = False) -> None:
    """Download StatsBomb match metadata (fast)."""
    from statsbombpy import sb

    STATSBOMB_DIR.mkdir(parents=True, exist_ok=True)
    competitions = sb.competitions()
    available = competitions[
        competitions["competition_name"].str.lower().apply(
            lambda n: any(k in n for k in ["world cup", "euro", "copa america", "africa cup"])
            and "women" not in n
        )
    ]

    for _, row in available.iterrows():
        comp_id = int(row["competition_id"])
        season_id = int(row["season_id"])
        comp_name = row.get("competition_name", str(comp_id))
        season_name = row.get("season_name", str(season_id))
        out_dir = STATSBOMB_DIR / str(comp_id) / str(season_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        matches_file = out_dir / "matches.json"

        if matches_file.exists() and not force:
            print(f"  [skip matches] {comp_name} {season_name}")
            continue

        print(f"  Downloading matches: {comp_name} {season_name}...")
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            matches_file.write_text(
                json.dumps(matches.to_dict(orient="records"), default=str),
                encoding="utf-8",
            )
            print(f"    Saved {len(matches)} matches")
        except Exception as e:
            print(f"    Warning: failed {comp_name} {season_name}: {e}")


def download_statsbomb_events(force: bool = False) -> None:
    """Download StatsBomb events per match (resumable, slower)."""
    from statsbombpy import sb

    for comp_dir in sorted(STATSBOMB_DIR.iterdir()):
        if not comp_dir.is_dir():
            continue
        for season_dir in sorted(comp_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            matches_file = season_dir / "matches.json"
            if not matches_file.exists():
                continue

            events_dir = season_dir / "events_by_match"
            events_dir.mkdir(exist_ok=True)

            with open(matches_file, encoding="utf-8") as f:
                matches = json.load(f)

            total = len(matches)
            for i, m in enumerate(matches, 1):
                mid = m["match_id"]
                match_events_file = events_dir / f"{mid}.json"
                if match_events_file.exists() and not force:
                    continue

                print(f"    Events {i}/{total} match_id={mid}...", end="\r")
                try:
                    events = sb.events(match_id=mid)
                    records = events.to_dict(orient="records")
                    match_events_file.write_text(
                        json.dumps(records, default=str), encoding="utf-8"
                    )
                except Exception:
                    match_events_file.write_text("[]", encoding="utf-8")

            print(f"  Saved events for {total} matches in {events_dir}          ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download project data")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--statsbomb-only", action="store_true")
    parser.add_argument("--transfermarkt-only", action="store_true")
    parser.add_argument("--international-only", action="store_true")
    parser.add_argument("--tm-api-only", action="store_true", help="Only fetch Transfermarkt API squad values")
    parser.add_argument("--polymarket-only", action="store_true", help="Only fetch Polymarket odds")
    parser.add_argument("--api-football-only", action="store_true", help="Sync WC 2026 fixtures from API-Football")
    parser.add_argument("--matches-only", action="store_true", help="Skip event download")
    parser.add_argument("--events-only", action="store_true", help="Only download events")
    args = parser.parse_args()

    if args.events_only:
        print("Downloading StatsBomb events (resumable)...")
        download_statsbomb_events(force=args.force)
    elif args.tm_api_only:
        print("Downloading Transfermarkt API squad values...")
        download_transfermarkt_api(force=args.force)
    elif args.polymarket_only:
        print("Downloading Polymarket odds...")
        download_polymarket_odds(force=args.force)
    elif args.api_football_only:
        print("Syncing API-Football WC 2026 fixtures...")
        download_api_football_fixtures(force=args.force)
    elif args.international_only:
        print("Downloading international results...")
        download_international(force=args.force)
        print("Downloading odds...")
        download_odds(force=args.force)
        print("Downloading Transfermarkt API squad values...")
        download_transfermarkt_api(force=args.force)
        print("Downloading Polymarket odds...")
        download_polymarket_odds(force=args.force)
        print("Syncing API-Football fixtures...")
        download_api_football_fixtures(force=args.force)
    elif not args.transfermarkt_only:
        print("Downloading international results...")
        download_international(force=args.force)
        print("Downloading odds...")
        download_odds(force=args.force)
        if not args.matches_only:
            print("Downloading StatsBomb matches (enrichment)...")
            download_statsbomb_matches(force=args.force)

    if not args.statsbomb_only and not args.events_only and not args.international_only:
        if not args.polymarket_only and not args.tm_api_only and not args.api_football_only:
            print("Downloading Transfermarkt API squad values...")
            download_transfermarkt_api(force=args.force)
            print("Downloading Polymarket odds...")
            download_polymarket_odds(force=args.force)
            print("Syncing API-Football fixtures...")
            download_api_football_fixtures(force=args.force)
        if not args.polymarket_only and not args.tm_api_only and not args.api_football_only:
            print("Downloading legacy Transfermarkt CSVs (optional)...")
            download_transfermarkt(force=args.force)

    print("Done.")


if __name__ == "__main__":
    main()
