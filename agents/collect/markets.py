from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import MARKET_SOURCE, ODDS_API_KEY, MARKET_MIN_ROWS, DATA_DIR, _dbg
from .status import _upsert_status_market_source
from .cfbd_clients import CfbdClients
from .cache import ApiCache, get_odds_cache
from .odds_fanduel import get_market_lines_fanduel_for_weeks


def _cfbd_lines_to_bookstyle(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize CFBD lines to a book-style home-line spread
    if df.empty:
        return df
    out = df.copy()
    # If favorite is home, spread is positive for home; else negative
    def _home_line(row):
        try:
            sp = float(row.get("spread")) if row.get("spread") is not None else None
        except Exception:
            sp = None
        if sp is None:
            return None
        fav = str(row.get("favorite") or "").strip()
        home = str(row.get("home_team") or "").strip()
        if not fav or not home:
            return None
        return sp if fav == home else -sp

    out["market_spread_book"] = out.apply(_home_line, axis=1)
    out = out[["game_id", "week", "home_team", "away_team", "market_spread_book"]]
    return out


def get_market_lines_for_current_week(
    year: int, week: int, schedule_df: pd.DataFrame, apis: CfbdClients, cache: ApiCache
) -> pd.DataFrame:
    _dbg(f"get_market_lines_for_current_week: env MARKET_SOURCE={MARKET_SOURCE!r}, ODDS_API_KEY={'set' if bool(ODDS_API_KEY) else 'missing'}")
    _dbg(f"schedule rows={len(schedule_df)} requested week={week}")

    requested = (MARKET_SOURCE or "cfbd").lower()
    used = requested
    fb_reason = ""
    out_df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    # If FanDuel requested
    if requested == "fanduel":
        try:
            if not ODDS_API_KEY:
                used = "cfbd"
                fb_reason = "FanDuel requested but ODDS_API_KEY missing"
            else:
                weeks = list(range(1, int(week) + 1))
                fanduel_df, stats = get_market_lines_fanduel_for_weeks(year, weeks, schedule_df, get_odds_cache())
                # keep only <= current week
                fanduel_df = fanduel_df.loc[pd.to_numeric(fanduel_df["week"], errors="coerce") <= int(week)].copy()
                if len(fanduel_df) >= MARKET_MIN_ROWS:
                    out_df = fanduel_df
                    used = "fanduel"
                else:
                    used = "cfbd"
                    fb_reason = f"FanDuel available but returned too few rows ({len(fanduel_df)})"
        except Exception as e:
            used = "cfbd"
            fb_reason = f"FanDuel fetch error: {e}"

    # CFBD branch (either requested or fallback)
    if used != "fanduel":
        try:
            if not apis.lines_api:
                fb_reason = fb_reason or "CFBD lines API unavailable"
            else:
                lines_rows: List[pd.DataFrame] = []
                for w in range(1, int(week) + 1):
                    try:
                        ls = apis.lines_api.get_lines(year=year, week=w, season_type="both")
                        rows = []
                        for ln in ls or []:
                            game_id = getattr(ln, "id", None) or getattr(ln, "game_id", None)
                            home = getattr(ln, "home_team", None)
                            away = getattr(ln, "away_team", None)
                            fav = getattr(ln, "home_team", None) if getattr(ln, "home_favorite", False) else getattr(ln, "away_team", None)
                            spread = getattr(ln, "spread", None)
                            rows.append(
                                {
                                    "game_id": game_id,
                                    "week": w,
                                    "home_team": home,
                                    "away_team": away,
                                    "favorite": fav,
                                    "spread": spread,
                                }
                            )
                        dfl = pd.DataFrame(rows)
                        if not dfl.empty:
                            dfl = _cfbd_lines_to_bookstyle(dfl)
                            lines_rows.append(dfl)
                    except Exception:
                        continue
                if lines_rows:
                    out_df = pd.concat(lines_rows, ignore_index=True)
                else:
                    out_df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
        except Exception as e:
            fb_reason = fb_reason or f"CFBD fetch error: {e}"

    # Record status
    _upsert_status_market_source(used, requested, fb_reason, DATA_DIR)
    return out_df


__all__ = ["_cfbd_lines_to_bookstyle", "get_market_lines_for_current_week"]

