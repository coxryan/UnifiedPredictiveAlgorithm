from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .cfbd_clients import CfbdClients
from .cache import ApiCache
from .config import DATA_DIR, CACHE_ONLY, _dbg, REQUIRE_SCHED_MIN_ROWS
from agents.storage.sqlite_store import (
    read_table_from_path,
    read_named_table,
    write_named_table,
    delete_rows,
)
from agents.storage.sqlite_store import read_table_from_path


def discover_current_week(schedule: pd.DataFrame) -> Optional[int]:
    if "week" not in schedule or "date" not in schedule:
        return None
    # Use US/Pacific for week rollovers so Saturday games on the same date are considered "current"
    try:
        now = pd.Timestamp.now(tz="America/Los_Angeles").date()
    except Exception:
        now = pd.Timestamp.utcnow().date()
    sched = schedule.copy()
    try:
        sched["d"] = pd.to_datetime(sched["date"], errors="coerce").dt.date
    except Exception:
        return None
    # Include all games on or before today's date (Pacific) as "completed/current"
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

# ----------------------------------------------------------
# Schedule freshness/staleness check
# ----------------------------------------------------------
import datetime

def _is_schedule_stale(df: pd.DataFrame, year: int) -> bool:
    """
    Heuristic: a healthy season schedule from CFBD should include future dates well beyond 'today'.
    If the max date in the file is not at least within the next 2 days (Pacific), consider it stale.
    Also treat tiny schedules as stale if below REQUIRE_SCHED_MIN_ROWS when set (>0).
    """
    try:
        if df is None or df.empty:
            return True
        if REQUIRE_SCHED_MIN_ROWS and len(df) < REQUIRE_SCHED_MIN_ROWS:
            _dbg(f"schedule stale: too few rows ({len(df)} < {REQUIRE_SCHED_MIN_ROWS})")
            return True
        dts = pd.to_datetime(df.get("date"), errors="coerce")
        if hasattr(dts, "dt") and getattr(dts.dt, "tz", None) is not None:
            try:
                dts = dts.dt.tz_convert("America/Los_Angeles")
            except Exception:
                try:
                    dts = dts.dt.tz_localize("America/Los_Angeles")
                except Exception:
                    pass
        if dts.isna().all():
            _dbg("schedule stale: all dates NaT")
            return True
        max_d = dts.dropna().max()
        try:
            today_pt = pd.Timestamp.now(tz="America/Los_Angeles").date()
        except Exception:
            today_pt = pd.Timestamp.utcnow().date()

        max_date = None
        if isinstance(max_d, pd.Timestamp):
            try:
                if max_d.tzinfo is not None:
                    max_d = max_d.tz_convert("America/Los_Angeles")
            except Exception:
                try:
                    max_d = max_d.tz_localize("America/Los_Angeles")
                except Exception:
                    pass
            max_date = max_d.date()
        elif pd.notna(max_d):
            try:
                max_date = pd.Timestamp(max_d).date()
            except Exception:
                max_date = None

        # Expect schedule to extend AT LEAST 2 days beyond 'today' during the season.
        if max_date and max_date < (today_pt + datetime.timedelta(days=2)):
            _dbg(
                "schedule stale: max date "
                f"{max_date} < today+2 {(today_pt + datetime.timedelta(days=2))}"
            )
            return True
        # Also ensure the schedule we loaded matches the requested year (guard bad CSV)
        if not dts.dt.year.fillna(year).astype(int).eq(int(year)).any():
            _dbg("schedule stale: dates do not match requested year")
            return True
    except Exception:
        return True
    return False


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

    # Try raw relational store first
    try:
        raw_sched = read_named_table("raw_cfbd_games")
        if not raw_sched.empty:
            df_raw = raw_sched.loc[raw_sched.get("season") == year].copy()
            if not df_raw.empty:
                df_raw["week"] = pd.to_numeric(df_raw.get("week"), errors="coerce")
                schedule_cols = [
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
                df = df_raw[schedule_cols].copy()
                if norm_fbs:
                    df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
                if not _is_schedule_stale(df, year):
                    cache.set(f"sched:{year}", df.to_dict(orient="records"))
                    return df
    except Exception:
        pass

    key = f"sched:{year}"
    ok, data = cache.get(key)
    if ok:
      df = pd.DataFrame(data)
      if norm_fbs:
          df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
      if _is_schedule_stale(df, year):
          _dbg("cache schedule deemed stale → ignoring and refetching")
      else:
          return df

    # Try reading existing CSV (created by prior builds)
    try:
        p = os.path.join(DATA_DIR, "cfb_schedule.csv")
        df = read_table_from_path(p)
        if not df.empty:
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
            if _is_schedule_stale(df, year):
                _dbg("cached schedule deemed stale → will fetch from CFBD API")
            else:
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
        if not df.empty:
            df_raw = df.copy()
            df_raw["season"] = year
            df_raw["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
            delete_rows("raw_cfbd_games", "season", year)
            write_named_table(df_raw, "raw_cfbd_games", if_exists="append")
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
