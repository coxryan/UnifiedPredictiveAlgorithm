from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set

import math
import re
import unicodedata
from types import SimpleNamespace

import pandas as pd

from .config import MARKET_SOURCE, ODDS_API_KEY, MARKET_MIN_ROWS, DATA_DIR, DEBUG_MARKET, _dbg
from .status import _upsert_status_market_source
from .cfbd_clients import CfbdClients
from .cache import ApiCache, get_odds_cache
from .odds_fanduel import get_market_lines_fanduel_for_weeks
from agents.storage import read_dataset, write_dataset as storage_write_dataset, delete_rows


_CFBD_PROVIDER_PRIORITY: Dict[str, int] = {
    "consensus": 0,
    "fanduel": 1,
    "fan duel": 1,
    "draftkings": 2,
    "draft kings": 2,
    "caesars": 3,
    "betmgm": 4,
    "bet mgm": 4,
}


def _normalize_team_key(name: Any) -> str:
    if name is None:
        return ""
    s = str(name)
    if not s:
        return ""
    # Strip accents and punctuation so CFBD/FanDuel naming variants match schedule entries.
    normalized = unicodedata.normalize("NFKD", s)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
    return " ".join(normalized.split())


def _parse_formatted_spread(formatted: Any) -> Tuple[str, Optional[float]]:
    if formatted is None:
        return "", None
    text = str(formatted).strip()
    if not text:
        return "", None
    if text.upper().endswith(" PK"):
        return text[:-3].strip(), 0.0
    if text.upper() in {"PK", "PICK"}:
        return "", 0.0

    # Capture the first numeric token (with sign) as the spread value.
    match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", text)
    if not match:
        return "", None
    spread_val: Optional[float]
    try:
        spread_val = float(match.group(0))
    except ValueError:
        spread_val = None
    team_label = text[: match.start()].strip(" -:@")
    return team_label, spread_val


def _cfbd_line_to_home_spread(home_team: str, away_team: str, line: Any) -> Optional[float]:
    home_key = _normalize_team_key(home_team)
    away_key = _normalize_team_key(away_team)

    label, parsed_val = _parse_formatted_spread(getattr(line, "formatted_spread", None))
    if parsed_val is not None:
        label_key = _normalize_team_key(label) if label else ""
        if label_key and label_key == home_key:
            return float(parsed_val)
        if label_key and label_key == away_key:
            return float(-parsed_val)

    spread_val = getattr(line, "spread", None)
    try:
        spread_val = float(spread_val)
    except (TypeError, ValueError):
        spread_val = math.nan
    if spread_val is None or (isinstance(spread_val, float) and math.isnan(spread_val)):
        return None

    # Determine favourite direction using moneylines when available.
    home_ml = getattr(line, "home_moneyline", None)
    away_ml = getattr(line, "away_moneyline", None)
    home_fav: Optional[bool] = None
    try:
        if home_ml is not None and away_ml is not None:
            home_ml_f = float(home_ml)
            away_ml_f = float(away_ml)
            if not math.isnan(home_ml_f) and not math.isnan(away_ml_f):
                if home_ml_f < away_ml_f:
                    home_fav = True
                elif away_ml_f < home_ml_f:
                    home_fav = False
    except (TypeError, ValueError):
        home_fav = None

    if home_fav is None:
        # Fall back to assuming the sign is already home-relative.
        home_fav = spread_val <= 0

    return float(spread_val) if home_fav else float(-spread_val)


def _select_cfbd_line(game: Any) -> Optional[Tuple[float, str]]:
    lines = getattr(game, "lines", None) or []
    if not lines:
        return None

    def _priority(item: Any) -> Tuple[int, int]:
        provider = _normalize_team_key(getattr(item, "provider", ""))
        pri = _CFBD_PROVIDER_PRIORITY.get(provider, 99)
        spread = getattr(item, "spread", None)
        valid = 0
        try:
            valid = 0 if spread is not None and not math.isnan(float(spread)) else 1
        except Exception:
            valid = 1
        return valid, pri

    for ln in sorted(lines, key=_priority):
        home_team = getattr(game, "home_team", "")
        away_team = getattr(game, "away_team", "")
        spread_val = _cfbd_line_to_home_spread(home_team, away_team, ln)
        if spread_val is not None:
            provider = getattr(ln, "provider", None)
            return float(spread_val), provider
    return None


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
    out = out[keep]
    if "spread" in out.columns:
        out = out.loc[out["spread"].notna()]
    return out


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
            sel = _select_cfbd_line(ln)
            if sel is None:
                continue
            spread_val, provider = sel
            rows.append(
                {
                    "game_id": getattr(ln, "id", None) or getattr(ln, "game_id", None),
                    "week": w,
                    "home_team": getattr(ln, "home_team", None),
                    "away_team": getattr(ln, "away_team", None),
                    "spread": spread_val,
                    "provider": provider,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    df = pd.DataFrame(rows)
    _dbg(f"_fetch_cfbd_lines: raw rows collected={len(df)}")
    return _normalize_market_df(df)


def _cfbd_lines_to_bookstyle(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize CFBD lines to a book-style home-line spread
    if df.empty:
        return df
    out = df.copy()
    if "market_spread_book" in out.columns:
        out["market_spread_book"] = pd.to_numeric(out["market_spread_book"], errors="coerce")
        cols = [c for c in ["game_id", "week", "home_team", "away_team", "market_spread_book"] if c in out.columns]
        return out[cols]

    def _home_line(row):
        line = SimpleNamespace(
            spread=row.get("spread"),
            formatted_spread=row.get("formatted_spread"),
            home_moneyline=row.get("home_moneyline"),
            away_moneyline=row.get("away_moneyline"),
        )
        return _cfbd_line_to_home_spread(row.get("home_team"), row.get("away_team"), line)

    out["market_spread_book"] = out.apply(_home_line, axis=1)
    out = out[["game_id", "week", "home_team", "away_team", "market_spread_book"]]
    return out


def _build_started_map(schedule_df: pd.DataFrame) -> Dict[int, bool]:
    if schedule_df is None or schedule_df.empty or "game_id" not in schedule_df.columns:
        return {}

    sched = schedule_df.copy()
    sched["game_id"] = pd.to_numeric(sched.get("game_id"), errors="coerce")
    sched = sched.loc[sched["game_id"].notna()].copy()
    if sched.empty:
        return {}
    sched["game_id"] = sched["game_id"].astype("int64")

    now_ts = pd.Timestamp.utcnow()
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    started: Dict[int, bool] = {}

    for _, row in sched.iterrows():
        gid = int(row["game_id"])
        kickoff = None
        for col in ["kickoff_utc", "start_time", "datetime", "start_date", "date"]:
            val = row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            kickoff = pd.to_datetime(val, utc=True, errors="coerce")
            if kickoff is not None and pd.notna(kickoff):
                break
        flag = False
        if kickoff is not None and pd.notna(kickoff):
            try:
                if kickoff.tzinfo is None:
                    kickoff = kickoff.tz_localize("UTC")
            except Exception:
                pass
            if kickoff <= now_ts:
                flag = True
        hp = row.get("home_points")
        ap = row.get("away_points")
        if pd.notna(hp) or pd.notna(ap):
            flag = True
        started[gid] = flag

    return started


def _persist_market_snapshot(
    table: str,
    season: int,
    new_rows: pd.DataFrame,
    started_map: Dict[int, bool],
) -> None:
    timestamp = pd.Timestamp.utcnow().isoformat()
    new_df = new_rows.copy() if new_rows is not None else pd.DataFrame()
    if not new_df.empty:
        new_df["game_id"] = pd.to_numeric(new_df.get("game_id"), errors="coerce")
        new_df = new_df.loc[new_df["game_id"].notna()].copy()
        new_df["game_id"] = new_df["game_id"].astype("int64")
        new_df["week"] = pd.to_numeric(new_df.get("week"), errors="coerce")
        new_df = new_df.loc[new_df["week"].notna()].copy()
        new_df["week"] = new_df["week"].astype("int64")
        new_df["season"] = season
        new_df["retrieved_at"] = timestamp

    existing_all = read_dataset(table)
    if existing_all.empty:
        existing_year = pd.DataFrame()
    else:
        existing_year = existing_all.loc[existing_all.get("season") == season].copy()

    if not existing_year.empty:
        existing_year["game_id"] = pd.to_numeric(existing_year.get("game_id"), errors="coerce")
        existing_year = existing_year.loc[existing_year["game_id"].notna()].copy()
        existing_year["game_id"] = existing_year["game_id"].astype("int64")
        existing_year["week"] = pd.to_numeric(existing_year.get("week"), errors="coerce")
        existing_year = existing_year.loc[existing_year["week"].notna()].copy()
        existing_year["week"] = existing_year["week"].astype("int64")

    if new_df.empty:
        final_year = existing_year.copy()
    else:
        freeze_ids: Set[int] = set()
        if not existing_year.empty:
            existing_ids = set(existing_year["game_id"].tolist())
            for gid in new_df["game_id"].tolist():
                if gid in existing_ids and started_map.get(int(gid), False):
                    freeze_ids.add(int(gid))

        preserve_rows = (
            existing_year.loc[existing_year["game_id"].isin(list(freeze_ids))].copy()
            if freeze_ids and not existing_year.empty
            else pd.DataFrame()
        )

        if freeze_ids:
            new_df = new_df.loc[~new_df["game_id"].isin(list(freeze_ids))].copy()

        update_weeks = sorted(set(new_df["week"].astype(int).tolist())) if not new_df.empty else []
        existing_keep = (
            existing_year.loc[~existing_year["week"].isin(update_weeks)].copy()
            if (not existing_year.empty and update_weeks)
            else existing_year.copy()
        )

        parts = [existing_keep]
        if not preserve_rows.empty:
            parts.append(preserve_rows)
        if not new_df.empty:
            parts.append(new_df)

        final_year = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not final_year.empty:
            final_year = final_year.drop_duplicates(
                subset=[col for col in ["game_id", "week"] if col in final_year.columns],
                keep="last",
            )
            final_year["season"] = season

    delete_rows(table, "season", season)
    if final_year is not None and not final_year.empty:
        storage_write_dataset(final_year, table, if_exists="append")


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
    fanduel_raw_debug = pd.DataFrame()
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
    if pd.notna(next_week) and next_week >= 1:
        try:
            weeks.append(int(next_week))
        except Exception:
            pass
    weeks = sorted({int(w) for w in weeks if pd.notna(w)})

    required_weeks: Set[int] = set()
    optional_weeks: Set[int] = set()
    if pd.notna(current_week):
        try:
            required_weeks.add(int(current_week))
        except Exception:
            pass
    if pd.notna(next_week):
        try:
            next_week_int = int(next_week)
        except Exception:
            next_week_int = None
        if next_week_int is not None and (not required_weeks or next_week_int not in required_weeks):
            optional_weeks.add(next_week_int)

    def _load_cached_lines(table: str) -> pd.DataFrame:
        cached = read_dataset(table)
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

    started_map = _build_started_map(schedule_df)

    # Attempt to use cached FanDuel lines first
    cached_fanduel = _load_cached_lines("raw_fanduel_lines")
    cached_fanduel_required_ok = True
    missing_required_weeks: List[int] = []
    missing_optional_weeks: List[int] = []
    cached_weeks: Set[int] = set()
    if not cached_fanduel.empty:
        cached_week_values = pd.to_numeric(cached_fanduel.get("week"), errors="coerce")
        if cached_week_values is not None:
            cached_weeks = {
                int(w) for w in cached_week_values.dropna().astype("int64")
            }
        if required_weeks:
            missing_required_weeks = sorted(w for w in required_weeks if w not in cached_weeks)
            cached_fanduel_required_ok = not missing_required_weeks
            if missing_required_weeks:
                market_extra["fanduel_cached_missing_weeks"] = missing_required_weeks
        if optional_weeks:
            missing_optional_weeks = sorted(w for w in optional_weeks if w not in cached_weeks)
            if missing_optional_weeks:
                market_extra["fanduel_cached_missing_optional_weeks"] = missing_optional_weeks
    else:
        if required_weeks:
            cached_fanduel_required_ok = False
            missing_required_weeks = sorted(required_weeks)
            _dbg(
                "get_market_lines_for_current_week: cached FanDuel missing required weeks "
                f"{missing_required_weeks}; will attempt refresh"
            )
    if (
        requested == "fanduel"
        and not cached_fanduel.empty
        and len(cached_fanduel) >= MARKET_MIN_ROWS
        and cached_fanduel_required_ok
    ):
        out_df = _normalize_market_df(cached_fanduel)
        fanduel_norm = out_df.copy()
        used = "fanduel"
        market_extra["fanduel_cached_rows"] = int(len(out_df))
        _dbg(f"get_market_lines_for_current_week: using cached FanDuel rows={len(out_df)}")
    else:
        if requested != "fanduel":
            out_df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    # If FanDuel requested
    if requested == "fanduel":
        try:
            if used != "fanduel" or out_df.empty:
                if not ODDS_API_KEY:
                    used = "cfbd"
                    fb_reason = "FanDuel requested but ODDS_API_KEY missing"
                    out_df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
                else:
                    fanduel_df, stats = get_market_lines_fanduel_for_weeks(year, weeks, schedule_df, get_odds_cache())
                    if isinstance(fanduel_df, pd.DataFrame):
                        fanduel_raw_debug = fanduel_df.copy()
                    else:
                        fanduel_raw_debug = pd.DataFrame()
                    market_extra = {
                        "market_raw": stats.get("raw", 0),
                        "market_mapped": stats.get("mapped", 0),
                        "market_unmatched": stats.get("unmatched", 0),
                    }
                    fanduel_df = fanduel_df.loc[pd.to_numeric(fanduel_df["week"], errors="coerce") <= next_week].copy()
                    fanduel_norm = _normalize_market_df(fanduel_df)
                    fanduel_rows = int(len(fanduel_norm))
                    market_extra["fanduel_rows"] = fanduel_rows
                    if fanduel_rows > 0:
                        out_df = fanduel_norm.copy()
                        used = "fanduel"
                        store_df = out_df.copy()
                        _persist_market_snapshot(
                            "raw_fanduel_lines",
                            year,
                            store_df,
                            started_map,
                        )
                        if fanduel_rows < MARKET_MIN_ROWS:
                            fb_reason = (
                                fb_reason
                                or f"FanDuel rows below threshold ({fanduel_rows} < {MARKET_MIN_ROWS}); supplementing with CFBD"
                            )
                    else:
                        used = "cfbd"
                        fb_reason = fb_reason or "FanDuel returned no usable rows"
                        out_df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
        except Exception as e:
            used = "cfbd"
            fb_reason = f"FanDuel fetch error: {e}"
            out_df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    # CFBD branch (either requested or fallback)
    if used != "fanduel":
        if out_df.empty:
            cached_cfbd = _load_cached_lines("raw_cfbd_lines")
            if not cached_cfbd.empty:
                out_df = _normalize_market_df(cached_cfbd)
                cfbd_norm = out_df.copy()
                used = "cfbd"
                market_extra["cfbd_rows"] = int(len(cfbd_norm))
        if out_df.empty:
            cfbd_df = _fetch_cfbd_lines(year, weeks, apis)
            cfbd_norm = _normalize_market_df(cfbd_df)
            market_extra["cfbd_rows"] = int(len(cfbd_norm))
            out_df = cfbd_norm.copy()
            if not cfbd_norm.empty:
                store_cfbd = cfbd_norm.copy()
                _persist_market_snapshot(
                    "raw_cfbd_lines",
                    year,
                    store_cfbd,
                    started_map,
                )
                used = "cfbd"
        _dbg(f"get_market_lines_for_current_week: CFBD-only rows={len(out_df)}")
        if out_df.empty and not fb_reason:
            fb_reason = "CFBD lines API unavailable or returned no rows"
    else:
        cfbd_df = _fetch_cfbd_lines(year, weeks, apis)
        cfbd_norm = _normalize_market_df(cfbd_df)
        market_extra["cfbd_rows"] = int(len(cfbd_norm))
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

        if "spread" in out_df.columns:
            tol = 1e-6
            source_series = pd.Series("unknown", index=out_df.index)
            if "market_spread_fanduel" in out_df.columns:
                fd_match = (
                    out_df["market_spread_fanduel"].notna()
                    & (out_df["spread"] - out_df["market_spread_fanduel"]).abs() <= tol
                )
                source_series.loc[fd_match] = "fanduel"
            if "market_spread_cfbd" in out_df.columns:
                cf_match = (
                    out_df["market_spread_cfbd"].notna()
                    & (out_df["spread"] - out_df["market_spread_cfbd"]).abs() <= tol
                )
                source_series.loc[(source_series == "unknown") & cf_match] = "cfbd"
            if "market_spread_cfbd" in out_df.columns:
                if "market_spread_fanduel" in out_df.columns:
                    fd_missing = out_df["market_spread_fanduel"].isna()
                else:
                    fd_missing = pd.Series(True, index=out_df.index)
                cf_only = out_df["market_spread_cfbd"].notna() & fd_missing
                source_series.loc[cf_only] = "cfbd"
            source_series.loc[out_df["spread"].isna()] = "none"
            out_df["market_spread_source"] = source_series
            market_extra["fanduel_games"] = int((source_series == "fanduel").sum())
            market_extra["cfbd_games"] = int((source_series == "cfbd").sum())

    _log_fanduel_nan_debug(out_df, fanduel_norm, fanduel_raw_debug, requested)

    # Record status
    _upsert_status_market_source(used, requested, fb_reason, DATA_DIR, extra=market_extra or None)
    return out_df


def _log_fanduel_nan_debug(
    out_df: pd.DataFrame,
    fanduel_norm: pd.DataFrame,
    fanduel_raw: pd.DataFrame,
    requested_source: str,
) -> None:
    if not DEBUG_MARKET or (requested_source or "").lower() != "fanduel":
        return

    if out_df is None or out_df.empty:
        return

    if "market_spread_fanduel" in out_df.columns:
        try:
            missing_mask = pd.to_numeric(out_df["market_spread_fanduel"], errors="coerce").isna()
        except Exception:
            missing_mask = pd.Series(True, index=out_df.index)
    else:
        # No FanDuel column present; treat all rows as missing to aid debugging.
        missing_mask = pd.Series(True, index=out_df.index)

    if not bool(missing_mask.any()):
        return

    cols = [c for c in ["game_id", "week", "home_team", "away_team", "market_spread_cfbd", "market_spread_book"] if c in out_df.columns]
    missing_rows = out_df.loc[missing_mask, cols]
    _dbg(
        f"get_market_lines_for_current_week: fanduel_missing rows={len(missing_rows)} (showing up to 10)"
    )

    for _, row in missing_rows.head(10).iterrows():
        detail: Dict[str, Any] = {
            "game_id": row.get("game_id"),
            "week": row.get("week"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "cfbd": row.get("market_spread_cfbd"),
            "book": row.get("market_spread_book"),
        }

        norm_match = _match_market_row(fanduel_norm, row)
        if not norm_match.empty:
            detail["fanduel_norm"] = norm_match.head(1).iloc[0].to_dict()
        elif fanduel_norm is not None and isinstance(fanduel_norm, pd.DataFrame) and not fanduel_norm.empty:
            detail["fanduel_norm"] = {"candidates": int(len(fanduel_norm))}
        else:
            detail["fanduel_norm"] = None

        raw_match = _match_market_row(fanduel_raw, row)
        if not raw_match.empty:
            detail["fanduel_raw"] = raw_match.head(1).iloc[0].to_dict()
        elif fanduel_raw is not None and isinstance(fanduel_raw, pd.DataFrame) and not fanduel_raw.empty:
            detail["fanduel_raw"] = {"candidates": int(len(fanduel_raw))}
        else:
            detail["fanduel_raw"] = None

        _dbg(f"get_market_lines_for_current_week: fanduel_missing_detail {detail}")


def _match_market_row(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    candidates = df

    def _normalize_numeric(series: pd.Series) -> pd.Series:
        try:
            return pd.to_numeric(series, errors="coerce")
        except Exception:
            return series

    gid = row.get("game_id")
    if pd.notna(gid) and "game_id" in candidates.columns:
        norm_gid = _normalize_numeric(candidates["game_id"])
        try:
            candidates = candidates.loc[norm_gid == float(gid)]
        except Exception:
            pass

    if candidates.empty and "week" in df.columns and pd.notna(row.get("week")):
        norm_week = _normalize_numeric(df["week"])
        try:
            candidates = df.loc[norm_week == float(row.get("week"))]
        except Exception:
            candidates = df.loc[norm_week == row.get("week")]

    def _match_team(col: str, value: Any, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or col not in frame.columns or pd.isna(value):
            return frame
        try:
            target = str(value).strip().casefold()
            return frame.loc[frame[col].astype(str).str.strip().str.casefold() == target]
        except Exception:
            return frame

    candidates = _match_team("home_team", row.get("home_team"), candidates)
    candidates = _match_team("away_team", row.get("away_team"), candidates)
    return candidates


__all__ = ["_cfbd_lines_to_bookstyle", "get_market_lines_for_current_week"]
