"""Date-window filters for WC 2026 fixtures."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd


def _local_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now().astimezone()
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc).astimezone()
    return now.astimezone()


def _to_local_dates(series: pd.Series, tz) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_convert(tz).dt.date


def split_today_tomorrow(
    fixtures: pd.DataFrame,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split upcoming fixtures into today and tomorrow buckets (local timezone)."""
    if fixtures.empty:
        return fixtures.copy(), fixtures.copy()

    now_local = _local_now(now)
    tz = now_local.tzinfo
    today = now_local.date()
    tomorrow = today + timedelta(days=1)

    df = fixtures.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")
    local_dates = _to_local_dates(df["match_date"], tz)

    upcoming = df[df["is_upcoming"] == True].copy()  # noqa: E712
    if upcoming.empty:
        return upcoming, upcoming.copy()

    upcoming["_local_date"] = local_dates.loc[upcoming.index]
    today_df = upcoming[upcoming["_local_date"] == today].drop(columns=["_local_date"])
    tomorrow_df = upcoming[upcoming["_local_date"] == tomorrow].drop(columns=["_local_date"])
    return today_df.reset_index(drop=True), tomorrow_df.reset_index(drop=True)


def filter_upcoming_window(
    fixtures: pd.DataFrame,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return upcoming fixtures on today or tomorrow (local date)."""
    today_df, tomorrow_df = split_today_tomorrow(fixtures, now=now)
    if today_df.empty and tomorrow_df.empty:
        return pd.DataFrame()
    return pd.concat([today_df, tomorrow_df], ignore_index=True)


def filter_played(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return finished fixtures sorted by date descending."""
    if fixtures.empty:
        return fixtures.copy()
    played = fixtures[fixtures["is_finished"] == True].copy()  # noqa: E712
    played["match_date"] = pd.to_datetime(played["match_date"], utc=True, errors="coerce")
    return played.sort_values("match_date", ascending=False).reset_index(drop=True)


def is_sync_stale(synced_at: str | None, now: datetime | None = None) -> bool:
    """True if last fixture sync was before local today."""
    if not synced_at:
        return True
    try:
        synced = pd.Timestamp(synced_at)
        if synced.tzinfo is None:
            synced = synced.tz_localize("UTC")
        synced_local = synced.tz_convert(_local_now(now).tzinfo).date()
        return synced_local < _local_now(now).date()
    except (TypeError, ValueError):
        return True
