"""Player photo fetcher.

Resolution order:
1. API-Football v3  (uses API_FOOTBALL_KEY from .env — free tier ~100 req/day)
2. Wikimedia / Wikipedia  (no key, fully free, open-license)
3. Returns None — caller should show a placeholder

All results are cached for the session so the same player is never
fetched twice from the network.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Optional

import requests
from dotenv import load_dotenv

# Load .env values so the module works when run outside Streamlit's launcher too
_ENV_PATH = None
_here = __file__
for _ in range(6):   # walk up until we find .env
    import pathlib
    candidate = pathlib.Path(_here).parent / ".env"
    if candidate.exists():
        _ENV_PATH = str(candidate)
        break
    _here = str(pathlib.Path(_here).parent.parent)

load_dotenv(_ENV_PATH, override=False)

_API_FOOTBALL_KEY  = os.getenv("API_FOOTBALL_KEY", "")
_API_FOOTBALL_BASE = os.getenv("API_FOOTBALL_BASE", "https://v3.football.api-sports.io")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "fifa-wc-analysis/1.0"})

# ---------------------------------------------------------------------------
# API-Football helper
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _api_football_photo(player_name: str) -> Optional[str]:
    """Search API-Football for the player and return their headshot URL."""
    if not _API_FOOTBALL_KEY:
        return None
    try:
        resp = _SESSION.get(
            f"{_API_FOOTBALL_BASE}/players",
            headers={"x-apisports-key": _API_FOOTBALL_KEY},
            params={"search": player_name},
            timeout=6,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = data.get("response", [])
        if not results:
            return None
        # Prefer the result whose name most closely matches
        name_lower = player_name.lower()
        for item in results:
            p = item.get("player", {})
            if name_lower in (p.get("name") or "").lower():
                photo = p.get("photo")
                if photo:
                    return photo
        # Fallback: just take the first result's photo
        photo = results[0].get("player", {}).get("photo")
        return photo or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Wikipedia / Wikimedia helper
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _wikipedia_photo(player_name: str) -> Optional[str]:
    """Return a Wikipedia thumbnail URL for the player (free, no key needed)."""
    try:
        resp = _SESSION.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "titles": player_name,
                "pithumbsize": 500,
                "redirects": 1,
            },
            timeout=6,
        )
        if resp.status_code != 200:
            return None
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail")
            if thumb:
                return thumb.get("source")
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_player_photo(
    player_name: str,
    *,
    prefer_wikipedia: bool = False,
    sleep_between_calls: float = 0.0,
) -> Optional[str]:
    """Return the best available photo URL for *player_name*, or None.

    Resolution order (default):
      1. API-Football  →  if that fails
      2. Wikipedia

    Set ``prefer_wikipedia=True`` to flip the order (useful when you know the
    API-Football quota is exhausted).
    """
    if sleep_between_calls:
        time.sleep(sleep_between_calls)

    if prefer_wikipedia:
        return _wikipedia_photo(player_name) or _api_football_photo(player_name)

    return _api_football_photo(player_name) or _wikipedia_photo(player_name)


def clear_cache() -> None:
    """Invalidate the in-process LRU cache (useful in tests or on reload)."""
    _api_football_photo.cache_clear()
    _wikipedia_photo.cache_clear()
