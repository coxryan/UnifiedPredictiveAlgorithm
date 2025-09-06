from __future__ import annotations
import os
import unicodedata
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from agents.lib.cache import ApiCache
from agents.lib.cfbd_clients import CfbdClients

# Controls for caching empty market pulls
TTL_EMPTY = int(os.environ.get("UPA_TTL_LINES_EMPTY", "600"))   # 10 minutes default
TTL_NONEMPTY = int(os.environ.get("UPA_TTL_LINES", "21600"))    # 6 hours default
FORCE_REFRESH = os.environ.get("UPA_FORCE_REFRESH", "0") == "1"


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # simplify punctuation and common aliases
    s = s.replace("'", "").replace("’", "").replace(".", "").replace(",", "").replace("-", " ")
    s = s.replace("st.", "st").replace("state univ", "state").replace("univ", "university")
    s = " ".join(s.split())
    return s


ALIASES = {
    "hawaii": {"hawaii", "hawaii rainbow warriors", "hawaii warriors", "hawaii rainbow"},
    "san jose state": {"san jose st", "sjsu"},
    "miami (oh)": {"miami oh", "miami ohio"},
    "umass": {"massachusetts", "umass amherst"},
    "connecticut": {"uconn"},
    "louisiana monroe": {"ulm", "la monroe", "louisiana monroe warhawks"},
    "texas san antonio": {"utsa", "tx san antonio"},
    "texas el paso": {"utep"},
    "southern methodist": {"smu"},
}
def _alias_key(s: str) -> str:
    n = _norm(s)
    for canon, alset in ALIASES.items():
        if n == canon or n in alset:
            return canon
    return n


def current_week_from_cfbd(apis: CfbdClients, season: int) -> int:
    # Prefer CFBD week endpoint; fallback to max in schedule if needed
    try:
        weeks = apis.games_api.get_calendar(year=season, season_type="regular")
        for w in weeks:
            if getattr(w, "first_game_start", None) and getattr(w, "last_game_start", None):
                # choose current by date bounds if the SDK provides.
                # If not available, use 'week' field with current flag when present.
                pass
    except Exception:
        pass
    # Fallback: derive from scheduled dates (simple, robust)
    sched = apis.get_schedule_df(season=season)  # cached inside CfbdClients
    if "week" in sched.columns and not sched.empty:
        # choose the smallest upcoming or the current max, simple heuristic
        return int(sched["week"].max())
    return 1


def _fetch_market_rows_for_week(apis: CfbdClients, season: int, week: int) -> pd.DataFrame:
    """
    Fetch market/lines for the specific week from your chosen source via CfbdClients.
    This function should be implemented inside CfbdClients (or here) to hit the API you use.
    We’ll wrap it with cache + FORCE_REFRESH.
    """
    cache: ApiCache = apis.cache
    key = ("market", season, week)
    if FORCE_REFRESH:
        cache.delete(key)
        print("[market] FORCE REFRESH: ignoring cached result")

    hit = cache.get(key)
    if hit is not None:
        return hit

    # ---- Replace this with your actual market source call ----
    # If you already have a helper in CfbdClients like apis.get_market_lines(season, week),
    # call it and shape to DataFrame with columns: game_id, week, home_team, away_team, spread (book line)
    try:
        df = apis.get_market_lines_df(season=season, week=week)  # expected helper in CfbdClients
        ttl = TTL_NONEMPTY if len(df) else TTL_EMPTY
        cache.set(key, df, ttl=ttl)
        return df
    except Exception as e:
        print(f"[market][warn] market fetch failed for week={week}: {e}")
        df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
        cache.set(key, df, ttl=TTL_EMPTY)
        return df


def build_schedule_with_market_current_week_only(
    apis: CfbdClients,
    season: int,
    current_week: int,
) -> pd.DataFrame:
    """
    Returns the full-season schedule DataFrame with these columns at minimum:
      game_id, week, date, away_team, home_team, neutral_site,
      market_spread_book, market_is_synthetic

    - Only the CURRENT WEEK will have real market lines joined (when available).
    - All NON-CURRENT weeks will have market_spread_book=0.0 and market_is_synthetic=True.
    - For CURRENT week rows that fail to join, market_spread_book stays NaN and market_is_synthetic=False.
    """
    sched = apis.get_schedule_df(season=season).copy()  # cached inside CfbdClients
    if sched.empty:
        return pd.DataFrame(columns=[
            "game_id","week","date","away_team","home_team","neutral_site",
            "market_spread_book","market_is_synthetic"
        ])

    # Normalize keys on schedule
    sched["home_norm"] = sched["home_team"].map(_alias_key)
    sched["away_norm"] = sched["away_team"].map(_alias_key)

    # Stamp defaults
    sched["market_spread_book"] = pd.NA
    sched["market_is_synthetic"] = False

    # Fetch + join for current week only
    mw = _fetch_market_rows_for_week(apis, season, current_week)
    m_raw = len(mw)
    if m_raw:
        mw = mw.copy()
        # normalize market side
        mw["home_norm"] = mw["home_team"].map(_alias_key)
        mw["away_norm"] = mw["away_team"].map(_alias_key)
        mw = mw.rename(columns={"spread": "market_spread_book"})[[
            "game_id","week","home_norm","away_norm","market_spread_book"
        ]]
        # Join
        sched = sched.merge(
            mw,
            how="left",
            on=["game_id","week","home_norm","away_norm"],
            suffixes=("","_mkt"),
        )
        # If join missed because game_id differs but teams match, try team-only join for this week
        missing_mask = sched["week"].eq(current_week) & sched["market_spread_book"].isna()
        if missing_mask.any():
            left = sched.loc[missing_mask, ["home_norm","away_norm"]]
            right = mw[["home_norm","away_norm","market_spread_book"]].drop_duplicates()
            rejoin = left.merge(right, how="left", on=["home_norm","away_norm"])
            sched.loc[missing_mask, "market_spread_book"] = rejoin["market_spread_book"].values

    joined = int(sched.loc[sched["week"].eq(current_week), "market_spread_book"].notna().sum())
    print(f"[market] fetched_raw={m_raw}; joined_to_schedule={joined}")

    # For all non-current weeks, stamp synthetic market = 0.0
    non_current_mask = sched["week"].ne(current_week)
    sched.loc[non_current_mask, "market_spread_book"] = 0.0
    sched.loc[non_current_mask, "market_is_synthetic"] = True

    # Optional: write debug snapshots
    debug_dir = Path("data") / "debug"
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        if m_raw:
            mw.head(200).to_csv(debug_dir / f"market_raw_w{current_week}.csv", index=False)
        sched.loc[sched["week"].eq(current_week) & sched["market_spread_book"].notna()]\
             .head(200).to_csv(debug_dir / f"market_joined_w{current_week}.csv", index=False)
    except Exception:
        pass

    return sched[[
        "game_id","week","date","away_team","home_team","neutral_site",
        "market_spread_book","market_is_synthetic"
    ] + [c for c in sched.columns if c not in {
        "game_id","week","date","away_team","home_team","neutral_site",
        "market_spread_book","market_is_synthetic","home_norm","away_norm"
    }]]