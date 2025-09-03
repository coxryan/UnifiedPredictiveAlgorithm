# agents/lib/market.py
import math
from datetime import datetime, timezone
import pandas as pd

from .cache import ApiCache, cached_call

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def _iso_date(dt_str):
    if not dt_str:
        return ""
    return str(dt_str)[:10]

def _fbs_vs_fbs(g) -> bool:
    return bool(getattr(g, "home_conference", None)) and bool(getattr(g, "away_conference", None))

def _line_to_home_perspective(line_obj) -> float:
    """
    Return home-perspective spread (negative = home favorite), using a robust median across available fields.
    """
    cand = []

    # home-side spreads
    for attr in ("home_spread", "homeSpread", "home_spread_open", "home_spread_close"):
        v = getattr(line_obj, attr, None)
        if v is not None:
            cand.append(_safe_float(v))

    # away-side spreads -> invert to home perspective
    for attr in ("away_spread", "awaySpread", "away_spread_open", "away_spread_close"):
        v = getattr(line_obj, attr, None)
        if v is not None:
            cand.append(-_safe_float(v))

    # generic field sometimes appears
    v = getattr(line_obj, "spread", None)
    if v is not None:
        cand.append(-_safe_float(v))

    # formatted text fallback
    fs = getattr(line_obj, "formattedSpread", None) or getattr(line_obj, "formatted_spread", None)
    if isinstance(fs, str) and fs.strip():
        try:
            parts = fs.replace("–", "-").split()
            if len(parts) >= 2:
                cand.append(_safe_float(parts[1]))
        except Exception:
            pass

    values = [v for v in cand if math.isfinite(v)]
    if not values:
        return float("nan")
    values.sort()
    return values[len(values)//2]

def infer_current_week(year: int, games_api, cache: ApiCache,
                       ttl_calendar=86400, ttl_games=31536000) -> int:
    """
    Prefer calendar; fall back to min future week / max historical week.
    """
    # Calendar (if available)
    try:
        cal, _ = cached_call(cache, "calendar",
                             {"fn": "get_calendar", "year": year},
                             ttl_calendar,
                             lambda: games_api.get_calendar(year=year))
        if cal:
            today = datetime.now(timezone.utc).date()
            for c in cal:
                s = _iso_date(getattr(c, "first_game_start", None) or getattr(c, "start_date", None))
                e = _iso_date(getattr(c, "last_game_start", None) or getattr(c, "end_date", None))
                if s and e:
                    try:
                        sd = datetime.fromisoformat(s).date()
                        ed = datetime.fromisoformat(e).date()
                        if sd <= today <= ed:
                            wk = getattr(c, "week", None) or getattr(c, "season_week", None)
                            if wk:
                                return int(wk)
                    except Exception:
                        pass
    except Exception:
        pass

    # Fallback via games
    games, _ = cached_call(cache, "games",
                           {"fn": "get_games", "year": year, "season_type": "both"},
                           ttl_games,
                           lambda: games_api.get_games(year=year, season_type="both"))
    if not games:
        return 1

    today = datetime.now(timezone.utc).date()
    week_dates = {}
    for g in games:
        wk = getattr(g, "week", None)
        dt = _iso_date(getattr(g, "start_date", None) or getattr(g, "start_time", None))
        if wk and dt:
            try:
                d = datetime.fromisoformat(dt).date()
                week_dates.setdefault(wk, []).append(d)
            except Exception:
                pass
    future_weeks = [wk for wk, dates in week_dates.items() if min(dates) >= today]
    if future_weeks:
        return min(future_weeks)
    return max(week_dates) if week_dates else 1

# ---------- Smart cache for lines: short TTL for empty; longer TTL for non-empty ----------

def fetch_lines_current_week_only(year: int, week: int, team: str, bet_api, cache: ApiCache,
                                  ttl_nonempty=21600, ttl_empty=1800):
    """
    Returns (lines_list, cache_status)
    cache_status ∈ {"hit_nonempty","hit_empty","miss_fetched_nonempty","miss_fetched_empty"}
    """
    key = {"fn": "get_lines", "year": int(year), "week": int(week), "team": team or ""}

    # Try non-empty bucket first
    v = cache.get("lines_week_team_nonempty", key, ttl_nonempty)
    if v is not None:
        return v, "hit_nonempty"

    # Then try empty bucket (short TTL)
    v = cache.get("lines_week_team_empty", key, ttl_empty)
    if v is not None:
        return v, "hit_empty"

    # Fetch
    try:
        res = bet_api.get_lines(year=year, week=int(week), team=team) or []
    except Exception:
        res = []

    if res:
        cache.set("lines_week_team_nonempty", key, res)
        return res, "miss_fetched_nonempty"
    else:
        cache.set("lines_week_team_empty", key, res)
        return res, "miss_fetched_empty"

def build_schedule_with_market_current_week_only(year: int, apis: dict, cache: ApiCache,
                                                 ttl_games=31536000, ttl_lines_nonempty=21600, ttl_lines_empty=1800):
    """
    Returns a full-season FBS vs FBS schedule with market lines filled ONLY for the current week.
    Columns: game_id, week, date, away_team, home_team, neutral_site, market_spread_book
    """
    games_api = apis["games"]
    bet_api = apis["betting"]

    # Full season (cached long)
    games, _ = cached_call(cache, "games",
                           {"fn": "get_games", "year": year, "season_type": "both"},
                           ttl_games,
                           lambda: games_api.get_games(year=year, season_type="both"))

    curr_week = infer_current_week(year, games_api, cache)

    rows = []
    for g in games or []:
        if not _fbs_vs_fbs(g):
            continue
        ht = getattr(g, "home_team", "") or ""
        at = getattr(g, "away_team", "") or ""
        wk = getattr(g, "week", None)
        date = _iso_date(getattr(g, "start_date", None) or getattr(g, "start_time", None))
        neutral = getattr(g, "neutral_site", False)

        market_home = float("nan")
        if wk == curr_week:
            lines, _ = fetch_lines_current_week_only(
                year, wk, ht, bet_api, cache,
                ttl_nonempty=ttl_lines_nonempty, ttl_empty=ttl_lines_empty
            )
            maybe = []
            for ln in (lines or []):
                for snap in getattr(ln, "lines", []) or []:
                    val = _line_to_home_perspective(snap)
                    if math.isfinite(val):
                        maybe.append(val)
            if maybe:
                maybe.sort()
                market_home = maybe[len(maybe)//2]

        rows.append({
            "game_id": getattr(g, "id", None),
            "week": wk,
            "date": date,
            "away_team": at,
            "home_team": ht,
            "neutral_site": "1" if bool(neutral) else "0",
            "market_spread_book": round(market_home, 1) if math.isfinite(market_home) else "",
        })

    df = pd.DataFrame(rows).sort_values(["week", "date", "away_team", "home_team"], ignore_index=True)
    return df, curr_week