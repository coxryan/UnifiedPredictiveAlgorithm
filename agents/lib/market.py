# agents/lib/market.py
import os, math
import pandas as pd
from datetime import datetime, timezone
from .cache import ApiCache, cached_call

def _safe_float(x):
    try: return float(x)
    except: return float("nan")

def _iso_date(dt_str):
    if not dt_str: return ""
    return str(dt_str)[:10]

def _fbs_vs_fbs(g) -> bool:
    return bool(getattr(g,"home_conference",None)) and bool(getattr(g,"away_conference",None))

def _line_to_home_perspective(line_obj, home_team: str, away_team: str):
    cand=[]
    for attr in ("home_spread","homeSpread","home_spread_open","home_spread_close"):
        v=getattr(line_obj,attr,None)
        if v is not None: cand.append(("home",_safe_float(v)))
    for attr in ("away_spread","awaySpread","away_spread_open","away_spread_close"):
        v=getattr(line_obj,attr,None)
        if v is not None: cand.append(("away->home",-_safe_float(v)))
    v=getattr(line_obj,"spread",None)
    if v is not None: cand.append(("generic->home",-_safe_float(v)))
    fs = getattr(line_obj,"formattedSpread",None) or getattr(line_obj,"formatted_spread",None)
    if isinstance(fs,str) and fs.strip():
        try:
            parts = fs.replace("–","-").split()
            if len(parts)>=2:
                sign=_safe_float(parts[1])
                # best-effort heuristic; keep median logic to damp variance
                cand.append(("fmt", sign))
        except: pass
    values=[v for _,v in cand if math.isfinite(v)]
    if not values: return float("nan")
    values.sort()
    return values[len(values)//2]

def infer_current_week(year: int, games_api, cache: ApiCache, ttl_calendar=86400, ttl_games=86400):
    """
    Try calendars first; fall back to nearest upcoming/ongoing game date.
    """
    # Try calendar endpoint (if available in cfbd)
    try:
        cal, _ = cached_call(cache, "calendar", {"fn":"get_calendar","year":year}, ttl_calendar,
                             lambda: games_api.get_calendar(year=year))
        # pick week where today is between start_date and end_date (inclusive)
        now = datetime.now(timezone.utc).date()
        for c in cal or []:
            s = str(getattr(c,"first_game_start","") or getattr(c,"start_date",""))[:10]
            e = str(getattr(c,"last_game_start","") or getattr(c,"end_date",""))[:10]
            if s and e:
                try:
                    sd = datetime.fromisoformat(s).date()
                    ed = datetime.fromisoformat(e).date()
                    if sd <= now <= ed:
                        return int(getattr(c,"week",None) or getattr(c,"season_week",None) or 1)
                except: pass
    except Exception:
        pass

    # Fallback: choose min week where game date >= today; else max week
    games, _ = cached_call(cache, "games", {"fn":"get_games","year":year,"season_type":"both"}, ttl_games,
                           lambda: games_api.get_games(year=year, season_type="both"))
    if not games: return 1
    today = datetime.now(timezone.utc).date()
    week_dates = {}
    for g in games:
        wk = getattr(g,"week",None)
        dt = _iso_date(getattr(g,"start_date",None) or getattr(g,"start_time",None))
        if wk and dt:
            try:
                d = datetime.fromisoformat(dt).date()
                week_dates.setdefault(wk, []).append(d)
            except: pass
    future_weeks = [wk for wk,dates in week_dates.items() if min(dates) >= today]
    if future_weeks: return min(future_weeks)
    return max(week_dates) if week_dates else 1

def build_schedule_with_market_current_week_only(year: int, apis: dict, cache: ApiCache,
                                                 ttl_games=31536000, ttl_lines=21600):
    games_api = apis["games"]; bet_api = apis["betting"]
    # full schedule (cached long) so we can output all weeks
    games, _ = cached_call(cache, "games", {"fn":"get_games","year":year,"season_type":"both"}, ttl_games,
                           lambda: games_api.get_games(year=year, season_type="both"))
    curr_week = infer_current_week(year, games_api, cache)

    rows=[]
    for g in games or []:
        if not _fbs_vs_fbs(g): continue
        ht, at = getattr(g,"home_team",""), getattr(g,"away_team","")
        wk = getattr(g,"week",None)
        date = _iso_date(getattr(g,"start_date",None) or getattr(g,"start_time",None))
        neutral = getattr(g,"neutral_site",False)

        market_home = float("nan")
        if wk == curr_week:
            # Only fetch lines for the current week
            key_lines = {"fn":"get_lines","year":year,"week":wk,"team":ht}
            def fetch_lines(): return bet_api.get_lines(year=year, week=wk, team=ht) or []
            lines, _ = cached_call(cache, "lines_week_team", key_lines, ttl_lines, fetch_lines)
            maybe=[]
            for ln in (lines or []):
                for snap in getattr(ln,"lines",[]) or []:
                    val = _line_to_home_perspective(snap, ht, at)
                    if math.isfinite(val): maybe.append(val)
            if maybe:
                maybe.sort()
                market_home = maybe[len(maybe)//2]

        rows.append({
            "game_id": getattr(g,"id",None),
            "week": wk, "date": date,
            "away_team": at, "home_team": ht,
            "neutral_site": "1" if bool(neutral) else "0",
            "market_spread_book": round(market_home,1) if math.isfinite(market_home) else ""
        })

    df = pd.DataFrame(rows).sort_values(["week","date","away_team","home_team"], ignore_index=True)
    return df, curr_week