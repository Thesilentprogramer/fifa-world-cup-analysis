"""Cross-source national team name normalization."""

from __future__ import annotations

from difflib import get_close_matches

INTERNATIONAL_TO_CANONICAL: dict[str, str] = {
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea, South": "South Korea",
    "Korea DPR": "North Korea",
    "Korea, North": "North Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czech Republic": "Czechia",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Republic of Congo": "Congo",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Republic of Ireland": "Ireland",
    "China PR": "China",
    "Cape Verde Islands": "Cape Verde",
    "Curaçao": "Curacao",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "UAE": "United Arab Emirates",
    "USSR": "Russia",
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Serbia and Montenegro": "Serbia",
    "Yugoslavia": "Serbia",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "Chinese Taipei": "Taiwan",
    "Northern Cyprus": "Cyprus",
    "São Tomé e Príncipe": "Sao Tome and Principe",
}

API_FOOTBALL_TO_CANONICAL: dict[str, str] = {
    **INTERNATIONAL_TO_CANONICAL,
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Congo": "Congo",
    "Congo DR": "DR Congo",
    "FYR Macedonia": "North Macedonia",
    "Republic of Ireland": "Ireland",
    "Türkiye": "Turkey",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czechia",
    "South Korea": "South Korea",
    "North Korea": "North Korea",
}

STATSBOMB_TO_CANONICAL: dict[str, str] = {**INTERNATIONAL_TO_CANONICAL}

TRANSFERMARKT_TO_CANONICAL: dict[str, str] = {
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "USA": "United States",
    "Cote d'Ivoire": "Ivory Coast",
    "Czech Republic": "Czechia",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "FYR Macedonia": "North Macedonia",
    "Congo DR": "DR Congo",
    "Cape Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Curacao": "Curacao",
}

CANONICAL_ALIASES: dict[str, str] = {}
for mapping in (INTERNATIONAL_TO_CANONICAL, STATSBOMB_TO_CANONICAL, TRANSFERMARKT_TO_CANONICAL, API_FOOTBALL_TO_CANONICAL):
    CANONICAL_ALIASES.update(mapping)


def normalize_team_name(name: str, source: str = "statsbomb") -> str:
    """Normalize a team name to canonical form."""
    if not name or not isinstance(name, str):
        return name

    cleaned = name.strip()
    if source == "transfermarkt":
        mapping = TRANSFERMARKT_TO_CANONICAL
    elif source == "api_football":
        mapping = API_FOOTBALL_TO_CANONICAL
    elif source == "international":
        mapping = INTERNATIONAL_TO_CANONICAL
    else:
        mapping = STATSBOMB_TO_CANONICAL

    if cleaned in mapping:
        return mapping[cleaned]
    if cleaned in CANONICAL_ALIASES:
        return CANONICAL_ALIASES[cleaned]
    return cleaned


def fuzzy_match_team(name: str, known_teams: list[str], cutoff: float = 0.85) -> str | None:
    """Return closest known team name if above cutoff similarity."""
    if name in known_teams:
        return name
    matches = get_close_matches(name, known_teams, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def build_team_registry(names: list[str]) -> dict[str, str]:
    """Map raw names to canonical names for all known variants."""
    registry: dict[str, str] = {}
    canonical_set = {normalize_team_name(n, "international") for n in names}
    for raw in names:
        canonical = normalize_team_name(raw, "international")
        registry[raw] = canonical
        registry[canonical] = canonical
    for alias, canonical in CANONICAL_ALIASES.items():
        if canonical in canonical_set:
            registry[alias] = canonical
    return registry
