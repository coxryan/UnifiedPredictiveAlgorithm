from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .cfbd_clients import CfbdClients
from .cache import ApiCache
from .config import DATA_DIR, CACHE_ONLY


def discover_current_week(schedule: pd.DataFrame) -> Optional[int]:
    if "week" not in schedule or "date" not in schedule:
        return None
    now = pd.Timestamp.now(tz="UTC").date()
    sched = schedule.copy()
    try:
        sched["d"] = pd.to_datetime(sched["date"], errors="coerce").dt.date
    except Exception:
        return None
    valid = sched.dropna(subset=["d"]).loc[lambda d: d["d"] <= now]
    if valid.empty:
        w = pd.to_numeric(sched["week"], errors="coerce").dropna()
        return int(w.min()) if not w.empty else None
    return int(valid["week"].max())


def _dummy_schedule(year: int) -> pd.DataFrame:
    data = [
        {
            "game_id": 1001,
            "week": 1,
            "date": f"{year}-08-24",
            "kickoff_utc": f"{year}-08-24T16:00:00Z",
            "away_team": "Iowa State",
            "home_team": "Kansas State",
            "neutral_site": 1,
        },
        {
            "game_id": 1002,
            "week": 1,
            "date": f"{year}-08-24",
            "kickoff_utc": f"{year}-08-24T23:00:00Z",
            "away_team": "Stanford",
            "home_team": "Hawai'i",
            "neutral_site": 0,
        },
        {
            "game_id": 2001,
            "week": 2,
            "date": f"{year}-08-30",
            "kickoff_utc": f"{year}-08-30T23:00:00Z",
            "away_team": "Akron",
            "home_team": "Wyoming",
            "neutral_site": 0,
        },
    ]
    return pd.DataFrame(data)


def _date_only(x) -> Optional[str]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            return x[:10] if x else None
        if hasattr(x, "date"):
            try:
                return x.date().isoformat()
            except Exception:
                pass
        dt = pd.to_datetime(x, errors="coerce")
        if pd.notna(dt):
            try:
                return dt.date().isoformat()
            except Exception:
                return None
    except Exception:
        return None
    return None


def _iso_datetime_str(x) -> Optional[str]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            return x
        if hasattr(x, "isoformat"):
            try:
                return x.isoformat()
            except Exception:
                pass
        dt = pd.to_datetime(x, errors="coerce")
        if pd.notna(dt):
            try:
                return dt.isoformat()
            except Exception:
                return None
    except Exception:
        return None
    return None


def load_schedule_for_year(
    year: int,
    apis: CfbdClients,
    cache: ApiCache,
    fbs_set: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    # Build FBS set if possible
    norm_fbs: Optional[set] = None
    if fbs_set is not None:
        norm_fbs = {str(t).strip() for t in fbs_set if str(t).strip()}
    elif apis.teams_api:
        try:
            fbs = apis.teams_api.get_fbs_teams(year=year)
            norm_fbs = {getattr(t, "school", None) for t in (fbs or []) if getattr(t, "school", None)}
        except Exception:
            norm_fbs = None

    key = f"sched:{year}"
    ok, data = cache.get(key)
    if ok:
        df = pd.DataFrame(data)
        if norm_fbs:
            df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
        return df

    # Try reading existing CSV (created by prior builds)
    try:
        p = os.path.join(DATA_DIR, "cfb_schedule.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "date" in df.columns:
                df = df[df["date"].astype(str).str.startswith(str(year))].copy()
            if "neutral_site" not in df.columns:
                df["neutral_site"] = 0
            if norm_fbs:
                df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
            keep = [
                "game_id",
                "week",
                "date",
                "kickoff_utc",
                "away_team",
                "home_team",
                "neutral_site",
                "home_points",
                "away_points",
            ]
            df = df[[c for c in keep if c in df.columns]]
            cache.set(key, df.to_dict(orient="records"))
            return df
    except Exception:
        pass

    # If no CFBD api, fall back to dummy when nothing else is present
    if not apis.games_api or CACHE_ONLY:
        df = _dummy_schedule(year)
        if norm_fbs:
            df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
        cache.set(key, df.to_dict(orient="records"))
        return df

    # Fetch via CFBD
    try:
        games = apis.games_api.get_games(year=year, season_type="both")
        rows = []
        for g in games or []:
            rows.append(
                {
                    "game_id": getattr(g, "id", None),
                    "week": getattr(g, "week", None),
                    "date": _date_only(getattr(g, "start_date", None)),
                    "kickoff_utc": _iso_datetime_str(getattr(g, "start_date", None)),
                    "away_team": getattr(g, "away_team", None),
                    "home_team": getattr(g, "home_team", None),
                    "neutral_site": 1 if getattr(g, "neutral_site", False) else 0,
                    "home_points": getattr(g, "home_points", None),
                    "away_points": getattr(g, "away_points", None),
                }
            )
        df = pd.DataFrame(rows)
        if norm_fbs:
            df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
        cache.set(key, df.to_dict(orient="records"))
        return df
    except Exception as e:
        print(f"[warn] schedule fetch failed: {e}")
        df = _dummy_schedule(year)
        if norm_fbs:
            df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
        cache.set(key, df.to_dict(orient="records"))
        return df


__all__ = [
    "discover_current_week",
    "_dummy_schedule",
    "_date_only",
    "_iso_datetime_str",
    "load_schedule_for_year",
]

