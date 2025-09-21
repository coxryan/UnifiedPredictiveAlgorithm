from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import MARKET_SOURCE, ODDS_API_KEY, MARKET_MIN_ROWS, DATA_DIR, _dbg
from .status import _upsert_status_market_source
from .cfbd_clients import CfbdClients
from .cache import ApiCache, get_odds_cache
from .odds_fanduel import get_market_lines_fanduel_for_weeks
from agents.storage.sqlite_store import read_named_table, write_named_table, delete_rows


def _normalize_market_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
    out = df.copy()
    if "market_spread_book" in out.columns:
        out = out.rename(columns={"market_spread_book": "spread"})
    for col in ("spread", "week", "game_id"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("home_team", "away_team"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    keep = [c for c in ["game_id", "week", "home_team", "away_team", "spread"] if c in out.columns]
    return out[keep]


def _combine_market_sources(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    prim = _normalize_market_df(primary).copy()
    if not prim.empty:
        prim["__source_priority"] = 0
    sec = _normalize_market_df(secondary).copy()
    if not sec.empty:
        sec["__source_priority"] = 1

    combined = pd.concat([prim, sec], ignore_index=True)
    if combined.empty:
        return combined

    combined["__spread_null"] = combined["spread"].isna().astype(int)
    combined.sort_values(by=["__spread_null", "__source_priority"], inplace=True, ignore_index=True)
    key_cols = [c for c in ["game_id", "week", "home_team", "away_team"] if c in combined.columns]
    combined = combined.drop_duplicates(subset=key_cols, keep="first")
    combined = combined.drop(columns=[col for col in ["__source_priority", "__spread_null"] if col in combined.columns])
    try:
        primary_non_null = int(prim["spread"].notna().sum()) if not prim.empty else 0
    except Exception:
        primary_non_null = 0
    try:
        secondary_non_null = int(sec["spread"].notna().sum()) if not sec.empty else 0
    except Exception:
        secondary_non_null = 0
    _dbg(f"_combine_market_sources: combined rows={len(combined)} primary_non_null={primary_non_null} secondary_non_null={secondary_non_null}")
    return combined


def _fetch_cfbd_lines(year: int, weeks: List[int], apis: CfbdClients) -> pd.DataFrame:
    if not apis.lines_api:
        _dbg("_fetch_cfbd_lines: lines_api unavailable; skipping CFBD odds fetch")
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    rows: List[Dict[str, Any]] = []
    for w in weeks:
        try:
            ls = apis.lines_api.get_lines(year=year, week=w, season_type="both")
        except Exception as exc:
            _dbg(f"_fetch_cfbd_lines: CFBD fetch error week={w}: {exc}")
            continue
        for ln in ls or []:
            rows.append(
                {
                    "game_id": getattr(ln, "id", None) or getattr(ln, "game_id", None),
                    "week": w,
                    "home_team": getattr(ln, "home_team", None),
                    "away_team": getattr(ln, "away_team", None),
                    "favorite": getattr(ln, "home_team", None) if getattr(ln, "home_favorite", False) else getattr(ln, "away_team", None),
                    "spread": getattr(ln, "spread", None),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    df = pd.DataFrame(rows)
    _dbg(f"_fetch_cfbd_lines: raw rows collected={len(df)}")
    df = _cfbd_lines_to_bookstyle(df)
    df = df.rename(columns={"market_spread_book": "spread"})
    return _normalize_market_df(df)


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
    fanduel_norm = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
    cfbd_norm = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
    market_extra: Dict[str, Any] = {}

    current_week = int(week)
    try:
        max_sched_week = int(pd.to_numeric(schedule_df.get("week"), errors="coerce").max())
    except Exception:
        max_sched_week = current_week
    next_week = current_week + 1
    if pd.notna(max_sched_week):
        next_week = min(next_week, max_sched_week)
    weeks = list(range(1, current_week + 1))
    if next_week not in weeks and next_week >= 1:
        weeks.append(next_week)
    weeks = sorted(set(int(w) for w in weeks if pd.notna(w)))

    def _load_cached_lines(table: str) -> pd.DataFrame:
        cached = read_named_table(table)
        if cached.empty:
            return cached
        cached = cached.loc[cached.get("season") == year].copy()
        if cached.empty:
            return cached
        cached["week"] = pd.to_numeric(cached.get("week"), errors="coerce")
        cached = cached.loc[cached["week"].notna()]
        cached = cached.loc[cached["week"] <= next_week]
        if cached.empty:
            return cached
        dup_keys = [c for c in ["game_id", "week", "home_team", "away_team"] if c in cached.columns]
        if dup_keys:
            cached = cached.sort_values(by=["retrieved_at"] if "retrieved_at" in cached.columns else dup_keys)
            cached = cached.drop_duplicates(subset=dup_keys, keep="last")
        return cached

    # Attempt to use cached FanDuel lines first
    cached_fanduel = _load_cached_lines("raw_fanduel_lines")
    if requested == "fanduel" and not cached_fanduel.empty and len(cached_fanduel) >= MARKET_MIN_ROWS:
        out_df = _normalize_market_df(cached_fanduel)
        used = "fanduel"
        _dbg(f"get_market_lines_for_current_week: using cached FanDuel rows={len(out_df)}")
    else:
        out_df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    # If FanDuel requested
    if requested == "fanduel":
        try:
            if used != "fanduel":
                if not ODDS_API_KEY:
                    used = "cfbd"
                    fb_reason = "FanDuel requested but ODDS_API_KEY missing"
                else:
                    fanduel_df, stats = get_market_lines_fanduel_for_weeks(year, weeks, schedule_df, get_odds_cache())
                    market_extra = {"market_raw": stats.get("raw", 0), "market_mapped": stats.get("mapped", 0), "market_unmatched": stats.get("unmatched", 0)}
                    fanduel_df = fanduel_df.loc[pd.to_numeric(fanduel_df["week"], errors="coerce") <= next_week].copy()
                    fanduel_norm = _normalize_market_df(fanduel_df)
                    if len(fanduel_norm) >= MARKET_MIN_ROWS:
                        out_df = fanduel_norm.copy()
                        used = "fanduel"
                        store_df = out_df.copy()
                        store_df["season"] = year
                        store_df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
                        delete_rows("raw_fanduel_lines", "season", year)
                        write_named_table(store_df, "raw_fanduel_lines", if_exists="append")
                    else:
                        used = "cfbd"
                        fb_reason = f"FanDuel available but returned too few rows ({len(fanduel_df)})"
        except Exception as e:
            used = "cfbd"
            fb_reason = f"FanDuel fetch error: {e}"

    # CFBD branch (either requested or fallback)
    if used != "fanduel":
        if out_df.empty:
            cached_cfbd = _load_cached_lines("raw_cfbd_lines")
            if not cached_cfbd.empty:
                out_df = _normalize_market_df(cached_cfbd)
                used = "cfbd"
        if out_df.empty:
            cfbd_df = _fetch_cfbd_lines(year, weeks, apis)
            cfbd_norm = _normalize_market_df(cfbd_df)
            out_df = cfbd_norm.copy()
            if not cfbd_norm.empty:
                store_cfbd = cfbd_norm.copy()
                store_cfbd["season"] = year
                store_cfbd["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
                delete_rows("raw_cfbd_lines", "season", year)
                write_named_table(store_cfbd, "raw_cfbd_lines", if_exists="append")
                used = "cfbd"
        _dbg(f"get_market_lines_for_current_week: CFBD-only rows={len(out_df)}")
        if out_df.empty and not fb_reason:
            fb_reason = "CFBD lines API unavailable or returned no rows"
    else:
        cfbd_df = _fetch_cfbd_lines(year, weeks, apis)
        cfbd_norm = _normalize_market_df(cfbd_df)
        if not cfbd_norm.empty:
            before_rows = len(out_df)
            out_df = _combine_market_sources(out_df, cfbd_norm)
            added = len(out_df) - before_rows
            market_extra["cfbd_fallback_added"] = added
            _dbg(f"get_market_lines_for_current_week: merged CFBD fallback rows added={added}")
        else:
            _dbg("get_market_lines_for_current_week: CFBD fallback empty")

    key_cols = [c for c in ["game_id", "week", "home_team", "away_team"] if c in out_df.columns]
    if key_cols:
        if not fanduel_norm.empty:
            fd = fanduel_norm[key_cols + ["spread"]].drop_duplicates(subset=key_cols, keep="last")
            fd = fd.rename(columns={"spread": "market_spread_fanduel"})
            out_df = out_df.merge(fd, on=key_cols, how="left")
        if not cfbd_norm.empty:
            cf = cfbd_norm[key_cols + ["spread"]].drop_duplicates(subset=key_cols, keep="last")
            cf = cf.rename(columns={"spread": "market_spread_cfbd"})
            out_df = out_df.merge(cf, on=key_cols, how="left")

    # Record status
    _upsert_status_market_source(used, requested, fb_reason, DATA_DIR, extra=market_extra or None)
    return out_df


__all__ = ["_cfbd_lines_to_bookstyle", "get_market_lines_for_current_week"]
