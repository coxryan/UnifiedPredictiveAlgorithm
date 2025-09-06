from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# =========================
# Config
# =========================
DATA_DIR = "data"
CACHE_DIR = ".cache_api"
CACHE_TTL_DAYS = int(os.environ.get("CACHE_TTL_DAYS", "90"))
# If set, do not hit any external APIs; use cache only (write empty on miss)
CACHE_ONLY = os.environ.get("CACHE_ONLY", "0").strip().lower() in ("1", "true", "yes")



# If set, we push to Google Sheets when SHEET_ID and GOOGLE_APPLICATION_CREDENTIALS are valid
ENABLE_SHEETS = False  # you can flip to True later
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
MARKET_SOURCE = os.environ.get("MARKET_SOURCE", "fanduel").strip().lower()

# Separate cache for odds aggregator so we can switch sources freely
ODDS_CACHE_DIR = os.environ.get("ODDS_CACHE_DIR", ".cache_odds")
ODDS_CACHE_TTL_DAYS = int(os.environ.get("ODDS_CACHE_TTL_DAYS", "2"))

# Debug: verbose market selection logging (set DEBUG_MARKET=1/true to enable)
DEBUG_MARKET = os.environ.get("DEBUG_MARKET", "0").strip().lower() in ("1", "true", "yes", "y")
def _dbg(msg: str) -> None:
    if DEBUG_MARKET:
        try:
            print(f"[debug-market] {msg}", file=sys.stderr)
        except Exception:
            pass

_odds_cache_singleton: Optional[ApiCache] = None

def get_odds_cache() -> ApiCache:
    global _odds_cache_singleton
    if _odds_cache_singleton is None:
        _odds_cache_singleton = ApiCache(root=ODDS_CACHE_DIR, days_to_live=ODDS_CACHE_TTL_DAYS)
    return _odds_cache_singleton


# =========================
# Status helpers
# =========================
from typing import Optional
def _upsert_status_market_source(
    market_used: str,
    market_requested: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    data_dir: str = DATA_DIR
) -> None:
    """
    Merge-update data/status.json with the selected market source and status fields.
    Always writes:
      - market_source_used: the used source (lowercased)
      - market_source: (back-compat) the used source (lowercased)
      - market_source_config: the requested source (lowercased, if provided)
      - market_fallback_reason: string if provided (else absent)
      - generated_at_utc: always refreshed to current UTC ISO timestamp
    Leaves other fields intact if present; creates the file if missing.
    """
    try:
        os.makedirs(data_dir, exist_ok=True)
        p = os.path.join(data_dir, "status.json")
        payload: Dict[str, Any]
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    payload = json.load(f) or {}
            except Exception:
                payload = {}
        else:
            payload = {}
        # Always set used/actual market (lowercased)
        used_lc = (market_used or "cfbd").strip().lower()
        payload["market_source_used"] = used_lc
        payload["market_source"] = used_lc  # back-compat
        # Requested/config market, if provided
        if market_requested is not None:
            payload["market_source_config"] = (market_requested or "").strip().lower()
        else:
            payload.pop("market_source_config", None)
        # Back-compat + aliases for UI
        if market_requested is not None:
            req_lc = (market_requested or "").strip().lower()
            payload["requested_market"] = req_lc
            payload["market_requested"] = req_lc
            payload["market_source_requested"] = req_lc
        else:
            payload.pop("requested_market", None)
            payload.pop("market_requested", None)
            payload.pop("market_source_requested", None)
        # Fallback reason, if provided
        if fallback_reason:
            payload["market_fallback_reason"] = str(fallback_reason)
        else:
            payload.pop("market_fallback_reason", None)
        if fallback_reason:
            payload["fallback_reason"] = str(fallback_reason)
        else:
            payload.pop("fallback_reason", None)
        # Always refresh timestamp for cache busting
        payload["generated_at_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        # non-fatal; never crash the collector for status write
        pass


# =========================
# Small file-backed cache
# =========================
class ApiCache:
    def __init__(self, root: str = CACHE_DIR, days_to_live: int = CACHE_TTL_DAYS):
        self.root = root
        self.ttl = days_to_live * 86400
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        sub = os.path.join(self.root, h[:2], h[2:4])
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{h}.json")

    def get(self, key: str) -> Tuple[bool, Any]:
        p = self._path(key)
        if not os.path.exists(p):
            return False, None
        try:
            if time.time() - os.path.getmtime(p) > self.ttl:
                return False, None
            with open(p, "r") as f:
                return True, json.load(f)
        except Exception:
            return False, None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        with open(p, "w") as f:
            json.dump(value, f)


# =========================
# CFBD clients (optional)
# =========================
try:
    import cfbd  # type: ignore
except Exception:
    cfbd = None


@dataclass
class CfbdClients:
    bearer_token: str
    teams_api: Any = None
    players_api: Any = None
    ratings_api: Any = None
    games_api: Any = None
    lines_api: Any = None
    rankings_api: Any = None

    def __post_init__(self):
        if cfbd and self.bearer_token:
            cfg = cfbd.Configuration(access_token=self.bearer_token)
            cli = cfbd.ApiClient(cfg)
            self.teams_api = cfbd.TeamsApi(cli)
            self.players_api = cfbd.PlayersApi(cli)
            self.ratings_api = cfbd.RatingsApi(cli)
            self.games_api = cfbd.GamesApi(cli)
            self.lines_api = cfbd.BettingApi(cli)
            self.rankings_api = cfbd.RankingsApi(cli)


# =========================
# Helpers
# =========================
def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Auto-fix/model-grade known outputs before writing
    try:
        if isinstance(df, pd.DataFrame):
            base = os.path.basename(path)
            if base in ("upa_predictions.csv", "backtest_predictions_2024.csv", "upa_predictions_2024_backtest.csv"):
                if '_apply_book_grades' in globals():
                    try:
                        df = _apply_book_grades(df.copy())
                    except Exception:
                        pass
    except Exception:
        pass
    df.to_csv(path, index=False)


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

# --- Book-style grading helper ----------------------------------------------
# Negative line = home favorite. Grade using final score + book line.
# Returns: "CORRECT", "INCORRECT", "P" (push) or "" if not gradeable.
def _grade_pick_result(pick_side, home_points, away_points, market_home_line) -> str:
    try:
        if pick_side is None or str(pick_side).strip() == "":
            return ""
        hp = float(home_points) if home_points is not None else float("nan")
        ap = float(away_points) if away_points is not None else float("nan")
        m  = float(market_home_line) if market_home_line is not None else float("nan")
        if not (np.isfinite(hp) and np.isfinite(ap) and np.isfinite(m)):
            return ""
        # book-style: home covers if (home - away + market) > 0
        adj = (hp - ap) + m
        if abs(adj) < 1e-9:
            return "P"
        cover_home = 1 if adj > 0 else -1
        ps = str(pick_side).upper()
        pick_home = ("HOME" in ps) or ("(HOME)" in ps)
        pick_away = ("AWAY" in ps) or ("(AWAY)" in ps)
        if pick_home:
            return "CORRECT" if cover_home > 0 else "INCORRECT"
        if pick_away:
            return "CORRECT" if cover_home < 0 else "INCORRECT"
        return ""
    except Exception:
        return ""

# Apply grading to a predictions/backtest DataFrame if required columns exist
# - Infers model pick if not present using edge or (model - market)
# - Leaves expected_result blank unless an expected pick is provided
def _apply_book_grades(df: pd.DataFrame) -> pd.DataFrame:
    # Must have scores and market
    req = {"home_points", "away_points", "market_spread_book"}
    if not req.issubset(df.columns):
        return df
    # Normalize numeric columns
    for c in ["home_points", "away_points", "market_spread_book", "edge_points_book", "model_spread_book", "expected_market_spread_book"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _norm_pick(s):
        if s is None:
            return ""
        t = str(s).strip().upper()
        if "AWAY" in t:
            return "AWAY"
        if "HOME" in t:
            return "HOME"
        return ""

    # Infer model pick if missing
    if "model_pick_side" in df.columns:
        model_pick = df["model_pick_side"].map(_norm_pick)
    else:
        # Use edge if available; else use model - market
        edge = df.get("edge_points_book")
        if edge is None:
            edge = pd.to_numeric(df.get("model_spread_book"), errors="coerce") - pd.to_numeric(df.get("market_spread_book"), errors="coerce")
        edge = pd.to_numeric(edge, errors="coerce")
        # By convention: edge = model - market; edge > 0 => value AWAY; else HOME
        model_pick = edge.apply(lambda e: "AWAY" if pd.notna(e) and e > 0 else ("HOME" if pd.notna(e) else ""))

    # Expected pick only if explicitly present
    expected_pick = df["expected_pick_side"].map(_norm_pick) if "expected_pick_side" in df.columns else pd.Series([""] * len(df), index=df.index)

    # Compute results row-wise
    df["model_result"] = [
        _grade_pick_result(p, hp, ap, m)
        for p, hp, ap, m in zip(model_pick, df["home_points"], df["away_points"], df["market_spread_book"])
    ]
    df["expected_result"] = [
        _grade_pick_result(p, hp, ap, m)
        for p, hp, ap, m in zip(expected_pick, df["home_points"], df["away_points"], df["market_spread_book"])
    ]
    return df


def _normalize_percent(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return x * 100.0 if x <= 1.0 else x


def _scale_0_100(series: pd.Series) -> pd.Series:
    """Scale a numeric series into [0, 100].
    Always returns float dtype (NaN for missing) so downstream .round() is safe.
    """
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    # If nothing numeric, return all-NaN float series to avoid object dtype
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series(50.0, index=s.index, dtype="float64")
    out = (s - mn) / (mx - mn) * 100.0
    return out.astype("float64")


def discover_current_week(schedule: pd.DataFrame) -> Optional[int]:
    if "week" not in schedule or "date" not in schedule:
        return None
    now = datetime.now(timezone.utc).date()
    sched = schedule.copy()
    try:
        sched["d"] = pd.to_datetime(sched["date"], errors="coerce").dt.date
    except Exception:
        return None
    valid = sched.dropna(subset=["d"])
    valid = valid.loc[valid["d"] <= now]
    if valid.empty:
        w = pd.to_numeric(sched["week"], errors="coerce").dropna()
        return int(w.min()) if not w.empty else None
    return int(valid["week"].max())


# =========================
# Schedule
# =========================
def _dummy_schedule(year: int) -> pd.DataFrame:
    data = [
        {"game_id": 1001, "week": 1, "date": f"{year}-08-24", "away_team": "Iowa State", "home_team": "Kansas State", "neutral_site": 1},
        {"game_id": 1002, "week": 1, "date": f"{year}-08-24", "away_team": "Stanford", "home_team": "Hawai'i", "neutral_site": 0},
        {"game_id": 2001, "week": 2, "date": f"{year}-08-30", "away_team": "Akron", "home_team": "Wyoming", "neutral_site": 0},
    ]
    return pd.DataFrame(data)


# Robustly coerce CFBD start_date (str or datetime) to 'YYYY-MM-DD' or None
def _date_only(x) -> Optional[str]:
    if x is None:
        return None
    try:
        # Common cases: string like '2025-08-28T19:00:00.000Z'
        if isinstance(x, str):
            return x[:10] if x else None
        # datetime-like (has .date())
        if hasattr(x, "date"):
            try:
                return x.date().isoformat()
            except Exception:
                pass
        # Fallback via pandas parser
        dt = pd.to_datetime(x, errors="coerce")
        if pd.notna(dt):
            try:
                return dt.date().isoformat()
            except Exception:
                return None
    except Exception:
        return None
    return None

# Preserve full ISO datetime string (UTC if provided), else None
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
    # Build a normalized FBS set if provided
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

    if not apis.games_api:
        df = _dummy_schedule(year)
        if norm_fbs:
            df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
        cache.set(key, df.to_dict(orient="records"))
        return df

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
        print(f"[warn] schedule fetch failed: {e}", file=sys.stderr)
        df = _dummy_schedule(year)
        if norm_fbs:
            df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
        cache.set(key, df.to_dict(orient="records"))
        return df


# =========================
# Team Inputs (RP + Talent + SOS + Portal)
# =========================
def build_team_inputs_datadriven(year: int, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    # fbs map for conference
    team_conf: Dict[str, str] = {}
    if apis.teams_api:
        try:
            fbs = apis.teams_api.get_fbs_teams(year=year)
            team_conf = {t.school: (t.conference or "FBS") for t in fbs}
        except Exception as e:
            print(f"[warn] fbs teams fetch failed: {e}", file=sys.stderr)

    # Returning Production
    rp_rows: List[Dict[str, Any]] = []
    if apis.players_api and team_conf:
        conferences = sorted({c for c in team_conf.values() if c})
        for conf in conferences:
            key = f"rp:{year}:{conf}"
            ok, data = cache.get(key)
            if ok:
                items = data
            else:
                try:
                    items = apis.players_api.get_returning_production(year=year, conference=conf)
                    # store minimally
                    serial = []
                    for it in items or []:
                        serial.append(
                            {
                                "team": getattr(it, "team", None),
                                "conference": getattr(it, "conference", None),
                                "overall": getattr(it, "overall", None),
                                "offense": getattr(it, "offense", None),
                                "defense": getattr(it, "defense", None),
                                "total_ppa": getattr(it, "total_ppa", None),
                                "total_offense_ppa": getattr(it, "total_offense_ppa", None),
                                "total_defense_ppa": getattr(it, "total_defense_ppa", None)
                                or getattr(it, "total_defensive_ppa", None),
                                "total_passing_ppa": getattr(it, "total_passing_ppa", None),
                                "total_rushing_ppa": getattr(it, "total_rushing_ppa", None),
                            }
                        )
                    cache.set(key, serial)
                    items = serial
                except Exception as e:
                    print(f"[warn] returning production fetch failed for {conf}: {e}", file=sys.stderr)
                    items = []

            for it in items or []:
                rp_rows.append(
                    {
                        "team": it.get("team"),
                        "conference": it.get("conference") or team_conf.get(it.get("team"), "FBS"),
                        "_overall": it.get("overall"),
                        "_offense": it.get("offense"),
                        "_defense": it.get("defense"),
                        "_ppa_tot": it.get("total_ppa"),
                        "_ppa_off": it.get("total_offense_ppa") or ((it.get("total_passing_ppa") or 0) + (it.get("total_rushing_ppa") or 0)),
                        "_ppa_def": it.get("total_defense_ppa"),
                    }
                )
    rp_df = pd.DataFrame(rp_rows).drop_duplicates(subset=["team"])
# Ensure numeric for PPA/RP columns (prevents object dtype during scaling)
    for _col in ["_overall", "_offense", "_defense", "_ppa_tot", "_ppa_off", "_ppa_def"]:
        if _col in rp_df.columns:
            rp_df[_col] = pd.to_numeric(rp_df[_col], errors="coerce").astype("float64")
    if not rp_df.empty:
        rp_df["wrps_offense_percent"] = rp_df["_offense"].map(_normalize_percent)
        rp_df["wrps_defense_percent"] = rp_df["_defense"].map(_normalize_percent)
        rp_df["wrps_overall_percent"] = rp_df["_overall"].map(_normalize_percent)

        if rp_df["wrps_overall_percent"].isna().all():
            rp_df["wrps_overall_percent"] = _scale_0_100(rp_df["_ppa_tot"]).round(1)
        if rp_df["wrps_offense_percent"].isna().all():
            rp_df["wrps_offense_percent"] = _scale_0_100(rp_df["_ppa_off"]).round(1)
        if rp_df["wrps_defense_percent"].isna().all():
            rp_df["wrps_defense_percent"] = _scale_0_100(rp_df["_ppa_def"]).round(1)

        rp_df["wrps_percent_0_100"] = pd.to_numeric(rp_df["wrps_overall_percent"], errors="coerce").astype("float64")
        rp_df["wrps_percent_0_100"] = rp_df["wrps_percent_0_100"].round(1)

    # Team Talent
    talent_df = pd.DataFrame({"team": [], "talent_score_0_100": []})
    if apis.teams_api:
        key = f"talent:{year}"
        ok, data = cache.get(key)
        if ok:
            df = pd.DataFrame(data)
        else:
            try:
                items = apis.teams_api.get_talent(year=year)
                df = pd.DataFrame([{"team": x.team, "talent": float(getattr(x, "talent", 0) or 0)} for x in items or []])
                cache.set(key, df.to_dict(orient="records"))
            except Exception as e:
                print(f"[warn] talent fetch failed: {e}", file=sys.stderr)
                df = pd.DataFrame()
        if not df.empty:
            mn, mx = df["talent"].min(), df["talent"].max()
            if mx == mn:
                df["talent_score_0_100"] = 50.0
            else:
                df["talent_score_0_100"] = ((df["talent"] - mn) / (mx - mn) * 100.0).round(1)
            talent_df = df[["team", "talent_score_0_100"]]

    # Current season SRS ratings → rank and rank-based score (0..100)
    srs_cur_df = pd.DataFrame({"team": [], "srs_rating": [], "srs_rank_1_133": [], "srs_score_0_100": []})
    if apis.ratings_api:
        key_srs_cur = f"srs:{year}:cur"
        ok_cur, srs_cur = cache.get(key_srs_cur)
        if not ok_cur:
            try:
                items = apis.ratings_api.get_srs(year=year)
                srs_cur = [{"team": x.team, "rating": float(getattr(x, "rating", 0) or 0)} for x in (items or [])]
                cache.set(key_srs_cur, srs_cur)
            except Exception as e:
                print(f"[warn] srs current fetch failed: {e}", file=sys.stderr)
                srs_cur = []
        if srs_cur:
            tmp = pd.DataFrame(srs_cur).rename(columns={"rating": "srs_rating"})
            if not tmp.empty:
                tmp["srs_rank_1_133"] = tmp["srs_rating"].rank(ascending=False, method="min").astype(int)
                N = float(tmp["srs_rank_1_133"].max()) if not tmp["srs_rank_1_133"].empty else 133.0
                # Convert rank (1 best) to 0..100 (100 best)
                tmp["srs_score_0_100"] = (1.0 - (tmp["srs_rank_1_133"] - 1) / N) * 100.0
                srs_cur_df = tmp[["team", "srs_rating", "srs_rank_1_133", "srs_score_0_100"]]

    # Previous season SOS rank via SRS-based average opponent rating
    prior = year - 1
    sos_df = pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})
    if apis.ratings_api and apis.games_api:
        key_srs = f"srs:{prior}"
        key_games = f"games:{prior}"
        ok_srs, srs_data = cache.get(key_srs)
        ok_g, g_data = cache.get(key_games)
        if not ok_srs:
            try:
                srs_items = apis.ratings_api.get_srs(year=prior)
                srs_df = pd.DataFrame([{"team": x.team, "rating": float(x.rating or 0)} for x in srs_items or []])
                cache.set(key_srs, srs_df.to_dict(orient="records"))
            except Exception as e:
                print(f"[warn] srs fetch failed: {e}", file=sys.stderr)
                srs_df = pd.DataFrame()
        else:
            srs_df = pd.DataFrame(srs_data)
        if not ok_g:
            try:
                g_items = apis.games_api.get_games(year=prior, season_type="both")
                g_df = pd.DataFrame(
                    [
                        {"home_team": getattr(g, "home_team", None), "away_team": getattr(g, "away_team", None)}
                        for g in g_items or []
                    ]
                )
                cache.set(key_games, g_df.to_dict(orient="records"))
            except Exception as e:
                print(f"[warn] games fetch failed (prior): {e}", file=sys.stderr)
                g_df = pd.DataFrame()
        else:
            g_df = pd.DataFrame(g_data)

        if not srs_df.empty and not g_df.empty:
            srs_map = dict(zip(srs_df["team"], srs_df["rating"]))
            from collections import defaultdict

            opps = defaultdict(list)
            for _, row in g_df.iterrows():
                ht, at = row.get("home_team"), row.get("away_team")
                if ht in srs_map and at in srs_map:
                    opps[ht].append(srs_map[at])
                    opps[at].append(srs_map[ht])
            rows = [{"team": t, "sos_value": sum(v) / len(v)} for t, v in opps.items() if v]
            df = pd.DataFrame(rows)
            if not df.empty:
                df["prev_season_sos_rank_1_133"] = df["sos_value"].rank(ascending=False, method="min").astype(int)
                sos_df = df[["team", "prev_season_sos_rank_1_133"]]

    # Transfer Portal (net)
    portal_df = pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})
    if apis.players_api:
        key = f"portal:{year}"
        ok, data = cache.get(key)
        if not ok:
            try:
                items = apis.players_api.get_transfer_portal(year=year)
                serial = []
                for p in items or []:
                    serial.append(
                        {
                            "to_team": getattr(p, "destination", None) or getattr(p, "to_team", None),
                            "from_team": getattr(p, "origin", None) or getattr(p, "from_team", None),
                            "rating": getattr(p, "rating", None),
                            "stars": getattr(p, "stars", None),
                        }
                    )
                cache.set(key, serial)
                data = serial
            except Exception as e:
                print(f"[warn] portal fetch failed: {e}", file=sys.stderr)
                data = []
        from collections import defaultdict

        incoming = defaultdict(int)
        outgoing = defaultdict(int)
        rating_in = defaultdict(float)
        rating_out = defaultdict(float)
        for p in data or []:
            to_team = p.get("to_team")
            from_team = p.get("from_team")
            rating = p.get("rating")
            stars = p.get("stars")
            try:
                val = float(rating) if rating is not None else (float(stars) if stars is not None else 1.0)
            except Exception:
                val = 1.0
            if to_team:
                incoming[to_team] += 1
                rating_in[to_team] += val
            if from_team:
                outgoing[from_team] += 1
                rating_out[from_team] += val
        all_teams = set(list(incoming.keys()) + list(outgoing.keys()))
        rows = []
        for t in all_teams:
            cnt_net = incoming[t] - outgoing[t]
            val_net = rating_in[t] - rating_out[t]
            rows.append({"team": t, "portal_net_count": cnt_net, "portal_net_value": val_net})
        dfp = pd.DataFrame(rows)
        if not dfp.empty:
            def scale(series):
                s = pd.to_numeric(series, errors="coerce")
                if s.notna().sum() == 0:
                    return pd.Series([50.0] * len(s), index=s.index)
                mn, mx = s.min(), s.max()
                if mx == mn:
                    return pd.Series([50.0] * len(s), index=s.index)
                return (s - mn) / (mx - mn) * 100.0
            dfp["portal_net_0_100"] = (0.5 * scale(dfp["portal_net_count"]) + 0.5 * scale(dfp["portal_net_value"])).round(1)
            portal_df = dfp[["team", "portal_net_0_100", "portal_net_count", "portal_net_value"]]

    # Merge inputs
    df = rp_df.merge(talent_df, on="team", how="left") if not rp_df.empty else pd.DataFrame()
    if df.empty and not talent_df.empty:
        df = talent_df.copy()
        df["team"] = df["team"]
    if not df.empty:
        df = df.merge(srs_cur_df, on="team", how="left")
        df = df.merge(sos_df, on="team", how="left")
        df = df.merge(portal_df, on="team", how="left")
    else:
        # Soft fallback (UI still works). You can remove this if you always have token.
        seed = [
            {"team": "Kansas State", "conference": "Big 12", "wrps_percent_0_100": 60, "talent_score_0_100": 68, "portal_net_0_100": 55},
            {"team": "Iowa State", "conference": "Big 12", "wrps_percent_0_100": 55, "talent_score_0_100": 60, "portal_net_0_100": 52},
            {"team": "Hawai'i", "conference": "MWC", "wrps_percent_0_100": 48, "talent_score_0_100": 40, "portal_net_0_100": 45},
            {"team": "Stanford", "conference": "ACC", "wrps_percent_0_100": 50, "talent_score_0_100": 62, "portal_net_0_100": 43},
            {"team": "Wyoming", "conference": "MWC", "wrps_percent_0_100": 57, "talent_score_0_100": 50, "portal_net_0_100": 49},
            {"team": "Akron", "conference": "MAC", "wrps_percent_0_100": 45, "talent_score_0_100": 38, "portal_net_0_100": 42},
        ]
        df = pd.DataFrame(seed)

    # Fill conference via team_conf if missing
    if "conference" not in df.columns or df["conference"].isna().any():
        if team_conf:
            df["conference"] = df["team"].map(team_conf).fillna(df.get("conference")).fillna("FBS")
        else:
            df["conference"] = df.get("conference", "FBS")

    # Ensure expected columns
    for col in [
        "team",
        "conference",
        "wrps_offense_percent",
        "wrps_defense_percent",
        "wrps_overall_percent",
        "wrps_percent_0_100",
        "talent_score_0_100",
        "srs_rating",
        "srs_rank_1_133",
        "srs_score_0_100",
        "prev_season_sos_rank_1_133",
        "portal_net_0_100",
        "portal_net_count",
        "portal_net_value",
    ]:
        if col not in df.columns:
            df[col] = None

    if "conference" in df.columns:
        df.sort_values(["conference", "team"], inplace=True, ignore_index=True)
    else:
        df.sort_values(["team"], inplace=True, ignore_index=True)

    return df



# =========================
# FanDuel via The Odds API (optional aggregator)
# =========================
def _odds_api_fetch_fanduel(year: int, weeks: List[int], cache: ApiCache) -> List[Dict[str, Any]]:
    """Return a flat list of {home_name, away_name, point_home_book, commence_time} using The Odds API.
    Requires env ODDS_API_KEY. We request FanDuel spreads only.
    """
    if not ODDS_API_KEY:
        return []
    _dbg(f"odds_api_fetch_fanduel: weeks={weeks}, cache_root={cache.root}")
    out: List[Dict[str, Any]] = []
    sport = "americanfootball_ncaaf"
    base = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
    for wk in sorted(set(int(w) for w in weeks if str(w).isdigit())):
        key = f"oddsapi:fanduel:{year}:{wk}"
        ok, cached = cache.get(key)
        _dbg(f"odds_api_fetch_fanduel: wk={wk} cache_ok={ok} cached_items={len(cached) if ok and cached is not None else 0}")
        if ok and cached is not None:
            out.extend(cached)
            continue
        if CACHE_ONLY:
            cache.set(key, [])
            continue
        try:
            url = base.format(sport=sport)
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "spreads",
                "bookmakers": "fanduel",
            }
            r = requests.get(url, params=params, timeout=25)
            r.raise_for_status()
            data = r.json() or []
            _dbg(f"odds_api_fetch_fanduel: wk={wk} http_ok items={len(data)}")
            rows: List[Dict[str, Any]] = []
            for game in data:
                bks = game.get("bookmakers") or []
                if not bks:
                    continue
                mks = next((m for m in (bks[0].get("markets") or []) if m.get("key") == "spreads"), None)
                if not mks:
                    continue
                outs = mks.get("outcomes") or []
                # We need home outcome to compute book-style line for home (negative if home favorite)
                # The Odds API's outcome names are the team names as shown by them; we capture both
                # and resolve to schedule names later.
                # We'll compute a home-style point if we can identify which outcome is home.
                # The API includes game["home_team"]/game["away_team"].
                g_home = game.get("home_team")
                g_away = game.get("away_team")
                # Find the home outcome
                out_home = next((o for o in outs if o.get("name") == g_home), None)
                if out_home is None:
                    # fallback: first outcome could be home; skip if uncertain
                    continue
                try:
                    point_home_book = float(out_home.get("point")) if out_home.get("point") is not None else None
                except Exception:
                    point_home_book = None
                if point_home_book is None:
                    continue
                rows.append({
                    "home_name": g_home,
                    "away_name": g_away,
                    "point_home_book": point_home_book,  # book-style: negative = home favorite
                    "commence_time": game.get("commence_time"),
                })
            cache.set(key, rows)
            out.extend(rows)
        except Exception as e:
            print(f"[warn] odds api fetch failed (wk {wk}): {e}", file=sys.stderr)
            cache.set(key, [])
    _dbg(f"odds_api_fetch_fanduel: total_flat_rows={len(out)}")
    return out

def _resolve_names_to_schedule(schedule_df: pd.DataFrame, name: str) -> Optional[str]:
    """Heuristic: map OddsAPI team name to schedule 'school' name using containment/cleaning."""
    if not name:
        return None
    s_name = str(name).strip()
    schools = set(str(x).strip() for x in schedule_df["home_team"].dropna().unique()) | set(
        str(x).strip() for x in schedule_df["away_team"].dropna().unique()
    )
    # exact
    if s_name in schools:
        return s_name
    # common normalizations
    low = s_name.lower()
    def norm(x: str) -> str:
        return x.lower().replace("&", "and").replace(" st.", " state").replace(" st ", " state ")
    # try containment either direction
    for sch in schools:
        if norm(sch) in norm(s_name) or norm(s_name) in norm(sch):
            return sch
    return None

def get_market_lines_fanduel_for_weeks(year: int, weeks: List[int], schedule_df: pd.DataFrame, cache: ApiCache) -> pd.DataFrame:
    raw = _odds_api_fetch_fanduel(year, weeks, cache)
    _dbg(f"get_market_lines_fanduel_for_weeks: raw games from odds api={len(raw)}")
    if not raw:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])  # book-style
    # Build index of schedule by (away,home) for ID/week
    idx = {}
    for _, row in schedule_df.iterrows():
        a = str(row.get("away_team") or "").strip()
        h = str(row.get("home_team") or "").strip()
        if a and h:
            idx[(a, h)] = {"game_id": row.get("game_id"), "week": row.get("week")}
    out_rows: List[Dict[str, Any]] = []
    unmatched = []
    for g in raw:
        h_name = _resolve_names_to_schedule(schedule_df, g.get("home_name"))
        a_name = _resolve_names_to_schedule(schedule_df, g.get("away_name"))
        if not h_name or not a_name:
            unmatched.append((g.get("home_name"), g.get("away_name")))
            continue
        meta = idx.get((a_name, h_name))
        if not meta:
            continue
        try:
            spread = float(g.get("point_home_book"))
        except Exception:
            continue
        out_rows.append({
            "game_id": meta.get("game_id"),
            "week": meta.get("week"),
            "home_team": h_name,
            "away_team": a_name,
            "spread": spread,  # book-style (negative = home favorite)
        })
    _dbg(f"get_market_lines_fanduel_for_weeks: mapped rows={len(out_rows)} unmatched={len(unmatched)}")
    if DEBUG_MARKET and unmatched[:5]:
        _dbg(f"unmatched sample (up to 5): {unmatched[:5]}")
    if not out_rows:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])  # book-style
    return pd.DataFrame(out_rows)

# =========================
# Market: current week only
# =========================
def get_market_lines_for_current_week(year: int, week: int, schedule_df: pd.DataFrame, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    """
    Use CFBD or FanDuel Lines API; fetch ALL weeks up to and including `week`.
    Output columns: game_id, week, home_team, away_team, spread (book-style, negative = home favorite)
    If unavailable, return empty df (collector will treat market=0 for future games without lines).
    Records status (used/requested/fallback) in status.json.
    """
    _dbg(f"get_market_lines_for_current_week: env MARKET_SOURCE={MARKET_SOURCE!r}, ODDS_API_KEY={'set' if bool(ODDS_API_KEY) else 'missing'}")
    _dbg(f"schedule rows={len(schedule_df)} requested week={week}")
    # 1) Record requested market
    requested = (MARKET_SOURCE or 'cfbd').lower()
    used = requested
    fb_reason = ''
    df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    # 3) If requested is fanduel, try FanDuel branch
    if requested == "fanduel":
        try:
            _dbg("fanduel branch: starting")
            if not ODDS_API_KEY:
                _dbg("fanduel branch: missing ODDS_API_KEY → will fallback to CFBD")
                used = "cfbd"
                fb_reason = "FanDuel requested but ODDS_API_KEY missing"
            else:
                try:
                    w_int = int(week)
                except Exception:
                    w_int = None
                weeks = []
                if w_int is not None:
                    weeks = sorted({int(x) for x in pd.to_numeric(schedule_df.get("week"), errors="coerce").dropna().astype(int) if int(x) <= w_int})
                else:
                    weeks = sorted({int(x) for x in pd.to_numeric(schedule_df.get("week"), errors="coerce").dropna().astype(int)})
                _dbg(f"fanduel branch: weeks considered (<= requested): {weeks}")
                fan_df = get_market_lines_fanduel_for_weeks(year, weeks, schedule_df, get_odds_cache())
                _dbg(f"fanduel branch: rows from odds api mapped={0 if fan_df is None else len(fan_df)}")
                if fan_df is not None and isinstance(fan_df, pd.DataFrame) and len(fan_df) >= 5:
                    df = fan_df
                else:
                    used = "cfbd"
                    fb_reason = "FanDuel returned insufficient rows"
        except Exception as e:
            used = "cfbd"
            fb_reason = f"FanDuel branch error: {e}"
    # 4) If used is cfbd, do CFBD logic
    if used == "cfbd":
        _dbg("cfbd branch: starting")
        if not apis.lines_api:
            df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
        else:
            try:
                w = int(week)
            except Exception:
                df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
            else:
                weeks = sorted({int(x) for x in pd.to_numeric(schedule_df.get("week"), errors="coerce").dropna().astype(int) if int(x) <= w})
                _dbg(f"cfbd branch: weeks considered (<= requested): {weeks}")
                all_rows = []
                for wk in weeks:
                    key = f"lines:{year}:{wk}"
                    ok, cached = cache.get(key)
                    if ok:
                        dfi = pd.DataFrame(cached)
                    else:
                        if CACHE_ONLY:
                            cache.set(key, [])
                            dfi = pd.DataFrame()
                        else:
                            try:
                                lines = apis.lines_api.get_lines(year=year, week=int(wk))
                            except Exception as e:
                                print(f"[warn] market lines fetch failed for {year} w{wk}: {e}", file=sys.stderr)
                                lines = []
                            rows = []
                            for ln in lines or []:
                                gid = getattr(ln, "id", None)
                                wv = getattr(ln, "week", None)
                                ht = getattr(ln, "home_team", None)
                                at = getattr(ln, "away_team", None)
                                spread = None
                                try:
                                    if hasattr(ln, "lines") and ln.lines:
                                        for l in ln.lines:
                                            if hasattr(l, "spread") and l.spread is not None:
                                                spread = float(l.spread)
                                                break
                                except Exception:
                                    pass
                                if spread is None:
                                    continue
                                rows.append({"game_id": gid, "week": wv, "home_team": ht, "away_team": at, "spread": spread})
                            dfi = pd.DataFrame(rows)
                            cache.set(key, dfi.to_dict(orient="records"))
                    if not dfi.empty:
                        all_rows.append(dfi)
                _dbg(f"cfbd branch: accumulated weekly frames={len(all_rows)}")
                if all_rows:
                    df = pd.concat(all_rows, ignore_index=True)
                else:
                    df = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])
    # 5) Write status and return
    try:
        _upsert_status_market_source(used, requested, fb_reason)
    except Exception:
        pass
    try:
        summary = {
            "requested": requested,
            "used": used,
            "fallback_reason": fb_reason,
            "rows_returned": int(len(df) if isinstance(df, pd.DataFrame) else 0),
            "odds_key_present": bool(ODDS_API_KEY),
        }
        _dbg(f"market summary: {summary}")
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, "market_debug.json"), "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass
    return df


# ==========================================================
# CI/Debug: Helper entry-point to force market status + debug write
# ==========================================================
def market_debug_entry(year: int, market_source: Optional[str] = None) -> None:
    """
    CI helper: load schedule and force a single pass of market selection so that
    status.json (requested/used/fallback) and data/market_debug.json are written.
    Safe no-op on error.
    """
    try:
        # allow override from caller
        global MARKET_SOURCE
        if market_source:
            MARKET_SOURCE = str(market_source).strip().lower() or MARKET_SOURCE

        bearer = os.environ.get("BEARER_TOKEN", "").strip()
        apis = CfbdClients(bearer_token=bearer)
        cache = ApiCache()

        # Load schedule (FBS-only filtering is handled inside)
        schedule_df = load_schedule_for_year(year=year, apis=apis, cache=cache)

        # Discover the "current" week (or fall back to min week in schedule)
        wk = discover_current_week(schedule_df)
        if wk is None:
            wk_series = pd.to_numeric(schedule_df.get("week"), errors="coerce").dropna()
            wk = int(wk_series.min()) if not wk_series.empty else 1

        # Run the market selection pass (this also writes status + market_debug)
        _ = get_market_lines_for_current_week(
            year=year,
            week=int(wk),
            schedule_df=schedule_df,
            apis=apis,
            cache=cache,
        )
    except Exception as _e:
        # Never fail CI because of debug helper
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(os.path.join(DATA_DIR, "market_debug.json"), "w") as f:
                json.dump(
                    {
                        "requested": MARKET_SOURCE,
                        "error": str(_e),
                        "odds_key_present": bool(ODDS_API_KEY),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass


# =========================
# Weekly Elo ranks (ratings) + Poll ranks (AP/Coaches)

def _elo_ranks(year: int, weeks: List[int], apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    """
    Fetch (and cache) Elo ratings by week and return a per-week rank and a 0..100 score.
    Schema: week, team, elo, elo_rank_1_133, elo_score_0_100
    """
    if not apis.ratings_api:
        return pd.DataFrame(columns=["week", "team", "elo", "elo_rank_1_133", "elo_score_0_100"])

    uniq_weeks = sorted({int(w) for w in weeks if w is not None and str(w).strip().isdigit()})
    all_rows: List[Dict[str, Any]] = []

    for wk in uniq_weeks:
        key = f"elo:{year}:{wk}"
        ok, data = cache.get(key)
        if not ok and CACHE_ONLY:
            # On cache-miss in CACHE_ONLY mode, write an empty list and continue
            cache.set(key, [])
            data = []
            ok = True

        if ok:
            rows = data or []
        else:
            try:
                items = apis.ratings_api.get_elo_ratings(year=year, week=int(wk))
            except Exception as e:
                print(f"[warn] elo fetch failed y{year} w{wk}: {e}", file=sys.stderr)
                items = []

            rows = []
            for it in items or []:
                team = getattr(it, "team", None) or getattr(it, "school", None)
                rating = getattr(it, "elo", None) or getattr(it, "rating", None)
                try:
                    rating = float(rating)
                except Exception:
                    rating = None
                if team is None or rating is None:
                    continue
                rows.append({"week": int(wk), "team": team, "elo": rating})

            cache.set(key, rows)

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(columns=["week", "team", "elo", "elo_rank_1_133", "elo_score_0_100"])

    # Per-week ranks (1 is best)
    df["elo_rank_1_133"] = df.groupby("week")["elo"].rank(ascending=False, method="min").astype(int)
    # Convert rank to a 0..100 score (100 best) using the max rank present per week
    max_rank = df.groupby("week")["elo_rank_1_133"].transform("max").astype("float64")
    df["elo_score_0_100"] = (1.0 - (df["elo_rank_1_133"].astype("float64") - 1.0) / max_rank) * 100.0
    return df[["week", "team", "elo", "elo_rank_1_133", "elo_score_0_100"]]


# Helper: cache AP/Coaches poll ranks per week
def _poll_ranks(year: int, weeks: List[int], apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    """
    Fetch (and cache) AP and Coaches poll ranks by week.
    Schema: week, team, ap_rank, coaches_rank
    """
    if not apis.rankings_api:
        return pd.DataFrame(columns=["week", "team", "ap_rank", "coaches_rank"])

    uniq_weeks = sorted({int(w) for w in weeks if w is not None and str(w).strip().isdigit()})
    out_rows: List[Dict[str, Any]] = []

    for wk in uniq_weeks:
        key = f"polls:{year}:{wk}"
        ok, data = cache.get(key)
        if not ok and CACHE_ONLY:
            cache.set(key, [])
            data = []
            ok = True

        if ok:
            rows = data or []
        else:
            try:
                wr_list = apis.rankings_api.get_rankings(year=year, week=int(wk))
            except Exception as e:
                print(f"[warn] polls fetch failed y{year} w{wk}: {e}", file=sys.stderr)
                wr_list = []

            rows = []
            for wr in wr_list or []:
                polls = getattr(wr, "polls", None) or []
                ap_map: Dict[str, int] = {}
                coaches_map: Dict[str, int] = {}
                for p in polls:
                    pname = (getattr(p, "poll", "") or "").lower()
                    ranks = getattr(p, "ranks", None) or []
                    for r in ranks:
                        team = getattr(r, "school", None) or getattr(r, "team", None)
                        rank = getattr(r, "rank", None)
                        try:
                            rank = int(rank)
                        except Exception:
                            rank = None
                        if not team or rank is None:
                            continue
                        if "coaches" in pname:
                            coaches_map[team] = rank
                        elif pname == "ap" or "associated" in pname or "press" in pname:
                            ap_map[team] = rank
                teams = set(list(ap_map.keys()) + list(coaches_map.keys()))
                for t in teams:
                    rows.append(
                        {
                            "week": int(wk),
                            "team": t,
                            "ap_rank": ap_map.get(t),
                            "coaches_rank": coaches_map.get(t),
                        }
                    )

            cache.set(key, rows)

        out_rows.extend(rows)

    df = pd.DataFrame(out_rows)
    if df.empty:
        return pd.DataFrame(columns=["week", "team", "ap_rank", "coaches_rank"])
    return df[["week", "team", "ap_rank", "coaches_rank"]]