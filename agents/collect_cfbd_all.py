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

# ======================================================
# Config / Env
# ======================================================
DATA_DIR = os.environ.get("DATA_DIR", "data").strip() or "data"

# Generic API cache (CFBD + other)
CACHE_DIR = os.environ.get("CFBD_CACHE_DIR", os.environ.get("CACHE_DIR", ".cache_cfbd"))
CACHE_TTL_DAYS = int(os.environ.get("CACHE_TTL_DAYS", "90"))

# If set, do not hit any external APIs; use cache only (write empty on miss)
CACHE_ONLY = os.environ.get("CACHE_ONLY", "0").strip().lower() in ("1", "true", "yes")

# Optional Google Sheets push (disabled by default)
ENABLE_SHEETS = False
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
MARKET_SOURCE = os.environ.get("MARKET_SOURCE", "fanduel").strip().lower()

# Separate cache for FanDuel (Odds API) so we can flip sources freely
ODDS_CACHE_DIR = os.environ.get("ODDS_CACHE_DIR", ".cache_odds")
ODDS_CACHE_TTL_DAYS = int(os.environ.get("ODDS_CACHE_TTL_DAYS", "2"))

# Require schedule to have at least N rows (guards against bad/missing reads)
REQUIRE_SCHED_MIN_ROWS = int(os.environ.get("REQUIRE_SCHED_MIN_ROWS", "0") or 0)

# Debug logging for market selection & matching
DEBUG_MARKET = os.environ.get("DEBUG_MARKET", "0").strip().lower() in ("1", "true", "yes", "y")
MARKET_MIN_ROWS = int(os.environ.get("MARKET_MIN_ROWS", "1"))  # minimum rows to consider a market usable


def _dbg(msg: str) -> None:
    if DEBUG_MARKET:
        try:
            print(f"[debug-market] {msg}", file=sys.stderr)
        except Exception:
            pass


# ======================================================
# Status helpers
# ======================================================
def _upsert_status_market_source(
    market_used: str,
    market_requested: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    data_dir: str = DATA_DIR,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Merge-update data/status.json with the selected market source and status fields.
    Always writes:
      - market_source_used: the used source (lowercased)
      - market_source: (back-compat) the used source (lowercased)
      - market_source_config/requested_market: requested source (lowercased, if provided)
      - fallback_reason/market_fallback_reason: string if provided
      - generated_at_utc: current UTC ISO timestamp
    Keeps other fields intact if present; creates the file if missing.
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

        used_lc = (market_used or "cfbd").strip().lower()
        payload["market_source_used"] = used_lc
        payload["market_source"] = used_lc  # back-compat

        if market_requested is not None:
            req_lc = (market_requested or "").strip().lower()
            payload["market_source_config"] = req_lc
            payload["requested_market"] = req_lc
            payload["market_requested"] = req_lc
            payload["market_source_requested"] = req_lc
        else:
            for k in ("market_source_config", "requested_market", "market_requested", "market_source_requested"):
                payload.pop(k, None)

        if fallback_reason:
            payload["market_fallback_reason"] = str(fallback_reason)
            payload["fallback_reason"] = str(fallback_reason)
        else:
            payload.pop("market_fallback_reason", None)
            payload.pop("fallback_reason", None)

        payload["generated_at_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        if extra:
            payload.update(extra)

        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        # never crash on status writing
        pass


# ======================================================
# Small file-backed cache
# ======================================================
class ApiCache:
    def __init__(self, root: str = CACHE_DIR, days_to_live: int = CACHE_TTL_DAYS):
        self.root = root
        self.ttl = max(1, int(days_to_live)) * 86400
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
            if (time.time() - os.path.getmtime(p)) > self.ttl:
                return False, None
            with open(p, "r") as f:
                return True, json.load(f)
        except Exception:
            return False, None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        with open(p, "w") as f:
            json.dump(value, f)


_odds_cache_singleton: Optional[ApiCache] = None


def get_odds_cache() -> ApiCache:
    global _odds_cache_singleton
    if _odds_cache_singleton is None:
        _odds_cache_singleton = ApiCache(root=ODDS_CACHE_DIR, days_to_live=ODDS_CACHE_TTL_DAYS)
    return _odds_cache_singleton


# ======================================================
# CFBD clients (optional)
# ======================================================
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


# ======================================================
# Helpers
# ======================================================
def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Mirror/grade for UI compatibility when writing predictions/backtest
    try:
        base = os.path.basename(path)
        if base in ("upa_predictions.csv", "backtest_predictions_2024.csv", "upa_predictions_2024_backtest.csv"):
            if "_mirror_book_to_legacy_columns" in globals():
                try:
                    df = _mirror_book_to_legacy_columns(df.copy())
                except Exception:
                    pass
            if "_apply_book_grades" in globals():
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
        m = float(market_home_line) if market_home_line is not None else float("nan")
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


def _apply_book_grades(df: pd.DataFrame) -> pd.DataFrame:
    # Must have scores and market
    req = {"home_points", "away_points", "market_spread_book"}
    if not req.issubset(df.columns):
        return df

    # Normalize numeric columns
    for c in [
        "home_points",
        "away_points",
        "market_spread_book",
        "edge_points_book",
        "model_spread_book",
        "expected_market_spread_book",
    ]:
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
        edge = df.get("edge_points_book")
        if edge is None:
            edge = pd.to_numeric(df.get("model_spread_book"), errors="coerce") - pd.to_numeric(
                df.get("market_spread_book"), errors="coerce"
            )
        edge = pd.to_numeric(edge, errors="coerce")
        # edge = model - market; edge > 0 ⇒ value AWAY; else HOME
        model_pick = edge.apply(lambda e: "AWAY" if pd.notna(e) and e > 0 else ("HOME" if pd.notna(e) else ""))

    expected_pick = (
        df["expected_pick_side"].map(_norm_pick)
        if "expected_pick_side" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )

    df["model_result"] = [
        _grade_pick_result(p, hp, ap, m)
        for p, hp, ap, m in zip(model_pick, df["home_points"], df["away_points"], df["market_spread_book"])
    ]
    df["expected_result"] = [
        _grade_pick_result(p, hp, ap, m)
        for p, hp, ap, m in zip(expected_pick, df["home_points"], df["away_points"], df["market_spread_book"])
    ]
    return df


def _mirror_book_to_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror FanDuel/“book” columns into legacy columns the UI expects."""
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df

        def _missing(col: str) -> bool:
            return (col not in df.columns) or pd.to_numeric(df[col], errors="coerce").isna().all()

        mappings = [
            ("market_spread_book", "market_spread"),
            ("market_spread_book", "market_h"),
            ("model_spread_book", "model_spread"),
            ("edge_points_book", "edge"),
            ("value_points_book", "value"),
            ("ev_percent_book", "ev_percent"),
            ("ev_bps_book", "ev_bps"),
            ("expected_market_spread_book", "expected_market_spread"),
        ]
        for src, dst in mappings:
            if src in df.columns and _missing(dst):
                df[dst] = df[src]
        return df
    except Exception:
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
    s = pd.to_numeric(series, errors="coerce").astype("float64")
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


# ======================================================
# Schedule
# ======================================================
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
            # ensure expected cols
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
        print(f"[warn] schedule fetch failed: {e}", file=sys.stderr)
        df = _dummy_schedule(year)
        if norm_fbs:
            df = df.loc[df["home_team"].isin(norm_fbs) & df["away_team"].isin(norm_fbs)].reset_index(drop=True)
        cache.set(key, df.to_dict(orient="records"))
        return df


# ======================================================
# Team Inputs (RP + Talent + SRS + SOS + Portal)
# ======================================================
def build_team_inputs_datadriven(year: int, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    # conference map
    team_conf: Dict[str, str] = {}
    if apis.teams_api:
        try:
            fbs = apis.teams_api.get_fbs_teams(year=year)
            team_conf = {t.school: (t.conference or "FBS") for t in fbs}
        except Exception as e:
            print(f"[warn] fbs teams fetch failed: {e}", file=sys.stderr)

    # Returning Production (Connelly via CFBD)
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
                        "_ppa_off": it.get("total_offense_ppa")
                        or ((it.get("total_passing_ppa") or 0) + (it.get("total_rushing_ppa") or 0)),
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

    # Current season SRS ratings → rank and rank-score
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
                tmp["srs_score_0_100"] = (1.0 - (tmp["srs_rank_1_133"] - 1) / N) * 100.0
                srs_cur_df = tmp[["team", "srs_rating", "srs_rank_1_133", "srs_score_0_100"]]

    # Prior season SOS (approx via avg opp SRS)
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
        df["conference"] = "FBS"
    if not df.empty:
        df = df.merge(srs_cur_df, on="team", how="left")
        df = df.merge(sos_df, on="team", how="left")
        df = df.merge(portal_df, on="team", how="left")
    else:
        # minimal seed to keep UI alive if token missing
        seed = [
            {"team": "Kansas State", "conference": "Big 12", "wrps_percent_0_100": 60, "talent_score_0_100": 68, "portal_net_0_100": 55},
            {"team": "Iowa State", "conference": "Big 12", "wrps_percent_0_100": 55, "talent_score_0_100": 60, "portal_net_0_100": 52},
            {"team": "Hawai'i", "conference": "MWC", "wrps_percent_0_100": 48, "talent_score_0_100": 40, "portal_net_0_100": 45},
            {"team": "Stanford", "conference": "ACC", "wrps_percent_0_100": 50, "talent_score_0_100": 62, "portal_net_0_100": 43},
            {"team": "Wyoming", "conference": "MWC", "wrps_percent_0_100": 57, "talent_score_0_100": 50, "portal_net_0_100": 49},
            {"team": "Akron", "conference": "MAC", "wrps_percent_0_100": 45, "talent_score_0_100": 38, "portal_net_0_100": 42},
        ]
        df = pd.DataFrame(seed)

    # Fill missing conference using team_conf
    if "conference" not in df.columns or df["conference"].isna().any():
        if team_conf:
            df["conference"] = df["team"].map(team_conf).fillna(df.get("conference")).fillna("FBS")
        else:
            df["conference"] = df.get("conference", "FBS")

    # Ensure all expected columns exist
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


# ======================================================
# FanDuel via The Odds API
# ======================================================
def _odds_api_fetch_fanduel(year: int, weeks: List[int], cache: ApiCache) -> List[Dict[str, Any]]:
    """
    Return list of {home_name, away_name, point_home_book, commence_time} using The Odds API.
    We cache one “slate” per UTC day; TTL controlled by ODDS_CACHE_TTL_DAYS.
    """
    if not ODDS_API_KEY:
        _dbg("odds_api_fetch_fanduel: missing ODDS_API_KEY")
        return []

    day_key = datetime.utcnow().strftime("%Y%m%d")
    key = f"oddsapi:fanduel:daily:{day_key}"
    ok, cached = cache.get(key)
    _dbg(f"odds_api_fetch_fanduel: cache_ok={ok} cached_items={len(cached) if ok and cached is not None else 0} key={key}")
    if ok and cached is not None:
        return list(cached)

    if CACHE_ONLY:
        cache.set(key, [])
        _dbg("odds_api_fetch_fanduel: CACHE_ONLY=1 and cache miss → empty cached")
        return []

    out: List[Dict[str, Any]] = []
    sport = "americanfootball_ncaaf"
    base = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

    try:
        url = base.format(sport=sport)
        agg: List[Dict[str, Any]] = []
        for page in range(1, 6):
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "spreads",
                "bookmakers": "fanduel",
                "oddsFormat": "american",
                "dateFormat": "iso",
                "page": page,
            }
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 404:
                _dbg(f"odds_api_fetch_fanduel: page {page} -> 404 (stop)")
                break
            r.raise_for_status()
            data = r.json() or []
            _dbg(f"odds_api_fetch_fanduel: page={page} items={len(data)}")
            if not data:
                break
            agg.extend(data)

        rows: List[Dict[str, Any]] = []
        for game in agg:
            bks = game.get("bookmakers") or []
            if not bks:
                continue
            mk = None
            for bk in bks:
                markets = bk.get("markets") or []
                mk = next((m for m in markets if m.get("key") == "spreads"), None)
                if mk:
                    break
            if not mk:
                continue
            outs = mk.get("outcomes") or []
            g_home = game.get("home_team")
            g_away = game.get("away_team")
            out_home = next((o for o in outs if o.get("name") == g_home), None)
            if out_home is None:
                continue
            try:
                point_home_book = float(out_home.get("point")) if out_home.get("point") is not None else None
            except Exception:
                point_home_book = None
            if point_home_book is None:
                continue
            rows.append(
                {
                    "home_name": g_home,
                    "away_name": g_away,
                    "point_home_book": point_home_book,
                    "commence_time": game.get("commence_time"),
                }
            )

        _dbg(f"odds_api_fetch_fanduel: total_items={len(agg)} usable_rows={len(rows)} (saving to cache key={key})")
        cache.set(key, rows)
        return rows
    except Exception as e:
        print(f"[warn] odds api fetch failed: {e}", file=sys.stderr)
        cache.set(key, [])
        return []


# ------------- Name resolution (FanDuel → schedule) -------------------------
import difflib


def _date_from_iso(s: Any) -> Optional[str]:
    if s is None:
        return None
    try:
        if isinstance(s, str) and len(s) >= 10:
            return s[:10]
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return str(dt.date())
    except Exception:
        return None
    return None


def _best_fuzzy_match(q_name: str, candidates: Iterable[str], normalize_fn) -> Tuple[Optional[str], float, str]:
    try:
        qn = normalize_fn(q_name or "")
        cand_norm = [(c, normalize_fn(c)) for c in candidates]
        best_c, best_s = None, 0.0
        for raw, cn in cand_norm:
            s = difflib.SequenceMatcher(None, qn, cn).ratio()
            if s > best_s:
                best_s, best_c = s, raw
        return best_c, float(best_s), qn
    except Exception:
        return None, 0.0, str(q_name or "")


def _resolve_names_to_schedule(schedule_df: pd.DataFrame, name: str) -> Optional[str]:
    """
    Map Odds API/FanDuel team name to schedule school name using robust normalization.
    Strategy:
      1) Exact match on normalized/aliased string
      2) Parenthetical-stripped exact match
      3) Direct alias table lookup (short → canonical)
      4) Acronym match (e.g., "LSU" → "Louisiana State")
      5) Token-set fuzzy containment/Jaccard with conservative threshold
    """
    if not name:
        return None

    # --- helpers -------------------------------------------------------------
    MASCOT_WORDS = {
        "bulldogs","wildcats","tigers","aggressors","agies","aggies","gators","longhorns","buckeyes","nittany","lions","nittany lions",
        "yellow","jackets","yellow jackets","demon","deacons","demon deacons","crimson","tide","crimson tide","redhawks","red hawks",
        "chippewas","huskies","zips","warhawks","cardinals","terrapins","razorbacks","trojans","bruins","gophers","badgers","cornhuskers",
        "rebels","utes","bearcats","cowboys","mountaineers","hurricanes","seminoles","sooners","volunteers","commodores",
        "panthers","wolfpack","falcons","eagles","golden eagles","golden","golden flashes","flashes","blazers","tar","heels","tar heels",
        "skyhawks","gamecocks","blue devils","blue","blue hens","scarlet knights","knights","rainbow warriors","warriors","rainbows",
        "rainbow","broncos","lancers","gaels","lions","rams","owls","spartans","tigers","tide","pirates","raiders","mean green",
        "anteaters","jaguars","trojans","minutemen","red wolves","hokies","uconn huskies","bulls","thundering herd","mustangs","cavaliers",
        "paladins","mocs","moccasins","mocsins","thunderbirds","mountaineers","phoenix","blue raiders","jayhawks","illini","aztecs",
        "redbirds","salukis","lumberjacks","cowgirls","cowboys","bears","mavericks","rivers","catamounts","governors","bengals",
        "buccaneers","runnin","runnin bulldogs","runnin' bulldogs","runnin-bulldogs","lobos","vandals","owls","golden hurricane",
        "scarlet","scarlet knights",
        # appended for FanDuel variants
        "midshipmen","dukes","bearkats","roadrunners","cardinal","cougars","knights"
    }

    def strip_diacritics(s: str) -> str:
        try:
            s2 = s.replace("ʻ", "'").replace("’", "'").replace("`", "'")
            return s2.encode("ascii", "ignore").decode("ascii", "ignore")
        except Exception:
            return s

    STOP_WORDS = {"university", "univ", "the", "of", "men's", "womens", "women's", "college", "st", "st.", "state", "and", "at", "amp", "amp;"}

    def drop_mascots(tokens: list[str]) -> list[str]:
        if not tokens:
            return tokens
        toks = tokens[:]
        i = 0
        out = []
        while i < len(toks):
            if i + 1 < len(toks) and f"{toks[i]} {toks[i+1]}" in MASCOT_WORDS and len(toks) > 2:
                i += 2
                continue
            if toks[i] in MASCOT_WORDS and len(toks) > 1:
                i += 1
                continue
            out.append(toks[i])
            i += 1
        return out if out else tokens

    def clean(s: str) -> str:
        s = strip_diacritics(s or "").lower().strip()
        s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
        s = s.replace(" st.", " state").replace(" st ", " state ")
        import re
        s = re.sub(r"[^a-z0-9() ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        toks = [t for t in s.split() if t not in STOP_WORDS]
        toks = drop_mascots(toks)
        return " ".join(toks)

    def no_paren(s: str) -> str:
        import re
        # Strip any parenthetical suffixes, e.g., "Miami (OH)" -> "Miami"
        return re.sub(r"\([^)]*\)", "", s).strip()

    def acronym_from(s: str) -> str:
        base = []
        for t in clean(s).split():
            if t in STOP_WORDS:
                continue
            if t == "a&m":
                base.extend(list("A&M"))
                continue
            base.append(t[0])
        return "".join([c for c in base if c.isalpha() or c == "&"]).upper()

    alias_map = {
        # short forms / book variants
        "pitt": "pittsburgh",
        "ole miss": "mississippi",
        "app state": "appalachian state",
        "la tech": "louisiana tech",
        "ul monroe": "louisiana monroe",
        "la monroe": "louisiana monroe",
        "ul lafayette": "louisiana",
        "louisiana lafayette": "louisiana",
        "western ky": "western kentucky",
        "san jose st": "san jose state",
        "sj state": "san jose state",
        "fresno st": "fresno state",
        "southern miss": "southern mississippi",
        "mass": "massachusetts",
        "uconn": "connecticut",
        "cal": "california",
        "penn st": "penn state",
        "southern cal": "southern california",
        "la lafayette": "louisiana",
        "nc st": "nc state",
        "ga tech": "georgia tech",
        "vt": "virginia tech",
        "uva": "virginia",
        "hawaii": "hawaii",
        "hawai'i": "hawaii",
        "hawaiʻi": "hawaii",
        "lsu": "louisiana state",
        "byu": "brigham young",
        "smu": "southern methodist",
        "tcu": "texas christian",
        "ucf": "central florida",
        "usf": "south florida",
        "utsa": "texas san antonio",
        "utep": "texas el paso",
        "unlv": "nevada las vegas",
        "umass": "massachusetts",
        "usc": "southern california",
        "fiu": "florida international",
        "fau": "florida atlantic",
        "miami fl": "miami",
        "miami fla": "miami",
        "miami oh": "miami (oh)",
        # school+mascot forms
        "air force falcons": "air force",
        "akron zips": "akron",
        "alabama crimson tide": "alabama",
        "appalachian state mountaineers": "appalachian state",
        "arizona wildcats": "arizona",
        "arizona state sun devils": "arizona state",
        "arkansas razorbacks": "arkansas",
        "army black knights": "army",
        "auburn tigers": "auburn",
        "ball state cardinals": "ball state",
        "baylor bears": "baylor",
        "boise state broncos": "boise state",
        "boston college eagles": "boston college",
        "brigham young cougars": "brigham young",
        "cal golden bears": "california",
        "california golden bears": "california",
        "central florida knights": "central florida",
        "central michigan chippewas": "central michigan",
        "charlotte 49ers": "charlotte",
        "cincinnati bearcats": "cincinnati",
        "clemson tigers": "clemson",
        "coastal carolina chanticleers": "coastal carolina",
        "colorado buffaloes": "colorado",
        "colorado state rams": "colorado state",
        "duke blue devils": "duke",
        "east carolina pirates": "east carolina",
        "eastern michigan eagles": "eastern michigan",
        "florida gators": "florida",
        "florida state seminoles": "florida state",
        "georgia bulldogs": "georgia",
        "georgia southern eagles": "georgia southern",
        "georgia state panthers": "georgia state",
        "georgia tech yellow jackets": "georgia tech",
        "hawaii rainbow warriors": "hawaii",
        "houston cougars": "houston",
        "illinois fighting illini": "illinois",
        "indiana hoosiers": "indiana",
        "iowa hawkeyes": "iowa",
        "iowa state cyclones": "iowa state",
        "james madison dukes": "james madison",
        "kansas jayhawks": "kansas",
        "kansas state wildcats": "kansas state",
        "kent state golden flashes": "kent state",
        "kentucky wildcats": "kentucky",
        "liberty flames": "liberty",
        "louisiana ragin cajuns": "louisiana",
        "louisiana lafayette ragin cajuns": "louisiana",
        "louisiana monroe warhawks": "louisiana monroe",
        "louisiana state tigers": "louisiana state",
        "louisville cardinals": "louisville",
        "marshall thundering herd": "marshall",
        "maryland terrapins": "maryland",
        "memphis tigers": "memphis",
        "miami hurricanes": "miami",
        "miami (oh) redhawks": "miami (oh)",
        "michigan wolverines": "michigan",
        "michigan state spartans": "michigan state",
        "middle tennessee blue raiders": "middle tennessee",
        "minnesota golden gophers": "minnesota",
        "mississippi rebels": "mississippi",
        "mississippi state bulldogs": "mississippi state",
        "missouri tigers": "missouri",
        "navy midshipmen": "navy",
        "nebraska cornhuskers": "nebraska",
        "nevada wolf pack": "nevada",
        "new mexico lobos": "new mexico",
        "new mexico state aggies": "new mexico state",
        "north carolina tar heels": "north carolina",
        "nc state wolfpack": "nc state",
        "north texas mean green": "north texas",
        "northern illinois huskies": "northern illinois",
        "northwestern wildcats": "northwestern",
        "notre dame fighting irish": "notre dame",
        "ohio bobcats": "ohio",
        "ohio state buckeyes": "ohio state",
        "oklahoma sooners": "oklahoma",
        "oklahoma state cowboys": "oklahoma state",
        "old dominion monarchs": "old dominion",
        "oregon ducks": "oregon",
        "oregon state beavers": "oregon state",
        "penn state nittany lions": "penn state",
        "pittsburgh panthers": "pittsburgh",
        "purdue boilermakers": "purdue",
        "rice owls": "rice",
        "rutgers scarlet knights": "rutgers",
        "san diego state aztecs": "san diego state",
        "san jose state spartans": "san jose state",
        "smu mustangs": "southern methodist",
        "south alabama jaguars": "south alabama",
        "south carolina gamecocks": "south carolina",
        "south florida bulls": "south florida",
        "southern miss golden eagles": "southern mississippi",
        "stanford cardinal": "stanford",
        "syracuse orange": "syracuse",
        "tcu horned frogs": "texas christian",
        "temple owls": "temple",
        "tennessee volunteers": "tennessee",
        "texas longhorns": "texas",
        "texas a and m": "texas a&m",
        "texas a m": "texas a&m",
        "texas a&m aggies": "texas a&m",
        "texas state bobcats": "texas state",
        "texas tech red raiders": "texas tech",
        "toledo rockets": "toledo",
        "troy trojans": "troy",
        "tulane green wave": "tulane",
        "tulsa golden hurricane": "tulsa",
        "uab blazers": "uab",
        "ucf knights": "central florida",
        "ucla bruins": "ucla",
        "unlv rebels": "nevada las vegas",
        "usc trojans": "southern california",
        "utah utes": "utah",
        "utah state aggies": "utah state",
        "utsa roadrunners": "texas san antonio",
        "virginia cavaliers": "virginia",
        "virginia tech hokies": "virginia tech",
        "wake forest demon deacons": "wake forest",
        "washington huskies": "washington",
        "washington state cougars": "washington state",
        "western kentucky hilltoppers": "western kentucky",
        "western michigan broncos": "western michigan",
        "wisconsin badgers": "wisconsin",
        "wyoming cowboys": "wyoming",
    }
    # extra FanDuel → schedule mappings we saw in logs
    alias_map.update({
        "arizona wildcats":"arizona",
        "arkansas razorbacks":"arkansas",
        "arkansas state red wolves":"arkansas state",
        "byu cougars":"brigham young",
        "boston college eagles":"boston college",
        "california golden bears":"california",
        "charlotte 49ers":"charlotte",
        "coastal carolina chanticleers":"coastal carolina",
        "colorado state rams":"colorado state",
        "east carolina pirates":"east carolina",
        "eastern michigan eagles":"eastern michigan",
        "houston cougars":"houston",
        "iowa state cyclones":"iowa state",
        "kansas state wildcats":"kansas state",
        "miami hurricanes":"miami",
        "miami (oh) redhawks":"miami (oh)",
        "mississippi state bulldogs":"mississippi state",
        "nc state wolfpack":"nc state",
        "navy midshipmen":"navy",
        "nevada wolf pack":"nevada",
        "north carolina tar heels":"north carolina",
        "oklahoma state cowboys":"oklahoma state",
        "old dominion monarchs":"old dominion",
        "san diego st aztecs":"san diego state",
        "san jose st spartans":"san jose state",
        "south carolina gamecocks":"south carolina",
        "stanford cardinal":"stanford",
        "uab blazers":"uab",
        "ucf knights":"central florida",
        "unlv rebels":"nevada las vegas",
        "usc trojans":"southern california",
        "utah utes":"utah",
        "utep miners":"texas el paso",
        "utsa roadrunners":"texas san antonio",
        "washington huskies":"washington",
        "western michigan broncos":"western michigan",
        "appalachian state mountaineers":"appalachian state",
        "virginia cavaliers":"virginia",
        "texas tech red raiders":"texas tech",
        "baylor bears":"baylor",
        "purdue boilermakers":"purdue",
        "air force falcons":"air force",
        "wyoming cowboys":"wyoming",
        "georgia bulldogs":"georgia",
        "georgia tech yellow jackets":"georgia tech",
        "boise state broncos":"boise state",
        "washington state cougars":"washington state"
    })

    # Allow external overrides
    try:
        for _pth in (os.path.join(DATA_DIR, "team_aliases.json"), "agents/team_aliases.json", "data/team_aliases.json"):
            if os.path.exists(_pth):
                with open(_pth, "r") as _f:
                    _extra = json.load(_f) or {}
                # Normalize keys/values via local clean()
                _norm_extra = {clean(str(k)): clean(str(v)) for k, v in _extra.items() if isinstance(k, str) and isinstance(v, str)}
                alias_map.update(_norm_extra)
                break
    except Exception:
        pass

    def alias(s: str) -> str:
        cs = clean(s)
        return alias_map.get(cs, cs)

    if "home_team" not in schedule_df.columns or "away_team" not in schedule_df.columns:
        return None

    # Build schedule lookup maps
    schools = set(str(x).strip() for x in schedule_df["home_team"].dropna().unique()) | set(
        str(x).strip() for x in schedule_df["away_team"].dropna().unique()
    )

    norm_map: Dict[str, str] = {}
    acro_map: Dict[str, str] = {}
    token_index: List[Tuple[str, set]] = []

    for sch in schools:
        can = alias(sch)
        norm_map[can] = sch
        ac = acronym_from(sch)
        if ac:
            acro_map[ac] = sch
        token_index.append((sch, set(can.split())))

    q_raw = name
    q_norm = alias(q_raw)

    # 1) direct normalized match
    if q_norm in norm_map:
        return norm_map[q_norm]

    # 2) parenthetical-stripped direct match
    q_np = no_paren(q_norm)
    if q_np in norm_map:
        return norm_map[q_np]

    # 3) alias on stripped form
    q_alias = alias(q_np)
    if q_alias in norm_map:
        return norm_map[q_alias]

    # 4) acronym
    q_acro = acronym_from(q_raw)
    if q_acro and q_acro in acro_map:
        return acro_map[q_acro]

    # 5) fuzzy
    q_tokens = set(q_alias.split())
    best_team, best_score = None, 0.0
    for sch, toks in token_index:
        if not toks:
            continue
        inter = len(q_tokens & toks)
        if inter == 0:
            continue
        union = len(q_tokens | toks)
        jacc = inter / float(union) if union else 0.0
        contain = 1.0 if (q_tokens.issubset(toks) or toks.issubset(q_tokens)) else 0.0
        score = jacc + 0.25 * contain
        if score > best_score:
            best_score, best_team = score, sch

    if best_team and best_score >= 0.40:
        return best_team
    return None


def _resolve_names_to_schedule_with_details(schedule_df: pd.DataFrame, name: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Detailed resolver: (resolved_school, detail_dict)."""
    detail: Dict[str, Any] = {"q_raw": name, "q_norm": None, "q_no_paren": None, "q_alias": None, "best_team": None, "best_score": None}
    if not name:
        return None, detail

    # Keep this logic in sync with _resolve_names_to_schedule
    MASCOT_WORDS = {
        "bulldogs","wildcats","tigers","aggressors","agies","aggies","gators","longhorns","buckeyes","nittany","lions","nittany lions",
        "yellow","jackets","yellow jackets","demon","deacons","demon deacons","crimson","tide","crimson tide","redhawks","red hawks",
        "chippewas","huskies","zips","warhawks","cardinals","terrapins","razorbacks","trojans","bruins","gophers","badgers","cornhuskers",
        "rebels","utes","bearcats","cowboys","mountaineers","hurricanes","seminoles","sooners","volunteers","commodores",
        "panthers","wolfpack","falcons","eagles","golden eagles","golden","golden flashes","flashes","blazers","tar","heels","tar heels",
        "skyhawks","gamecocks","blue devils","blue","blue hens","scarlet knights","knights","rainbow warriors","warriors","rainbows",
        "rainbow","broncos","lancers","gaels","lions","rams","owls","spartans","tigers","tide","pirates","raiders","mean green",
        "anteaters","jaguars","trojans","minutemen","red wolves","hokies","uconn huskies","bulls","thundering herd","mustangs","cavaliers",
        "paladins","mocs","moccasins","mocsins","thunderbirds","mountaineers","phoenix","blue raiders","jayhawks","illini","aztecs",
        "redbirds","salukis","lumberjacks","cowgirls","cowboys","bears","mavericks","rivers","catamounts","governors","bengals",
        "buccaneers","runnin","runnin bulldogs","runnin' bulldogs","runnin-bulldogs","lobos","vandals","owls","golden hurricane",
        "scarlet","scarlet knights",
        # appended for FanDuel variants
        "midshipmen","dukes","bearkats","roadrunners","cardinal","cougars","knights"
    }

    def strip_diacritics(s: str) -> str:
        try:
            s2 = s.replace("ʻ", "'").replace("’", "'").replace("`", "'")
            return s2.encode("ascii", "ignore").decode("ascii", "ignore")
        except Exception:
            return s

    STOP_WORDS = {"university", "univ", "the", "of", "men's", "womens", "women's", "college", "st", "st.", "state", "and", "at", "amp", "amp;"}

    def drop_mascots(tokens: list[str]) -> list[str]:
        if not tokens:
            return tokens
        toks = tokens[:]
        i = 0
        out = []
        while i < len(toks):
            if i + 1 < len(toks) and f"{toks[i]} {toks[i+1]}" in MASCOT_WORDS and len(toks) > 2:
                i += 2
                continue
            if toks[i] in MASCOT_WORDS and len(toks) > 1:
                i += 1
                continue
            out.append(toks[i])
            i += 1
        return out if out else tokens

    def clean(s: str) -> str:
        s = strip_diacritics(s or "").lower().strip()
        s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
        s = s.replace(" st.", " state").replace(" st ", " state ")
        import re
        s = re.sub(r"[^a-z0-9() ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        toks = [t for t in s.split() if t not in STOP_WORDS]
        toks = drop_mascots(toks)
        return " ".join(toks)

    def no_paren(s: str) -> str:
        import re
        return re.sub(r"\([^)]*\)", "", s).strip()

    def acronym_from(s: str) -> str:
        base = []
        for t in clean(s).split():
            if t in STOP_WORDS:
                continue
            if t == "a&m":
                base.extend(list("A&M"))
                continue
            base.append(t[0])
        return "".join([c for c in base if c.isalpha() or c == "&"]).upper()

    alias_map = {
        "pitt": "pittsburgh",
        "ole miss": "mississippi",
        "app state": "appalachian state",
        "la tech": "louisiana tech",
        "ul monroe": "louisiana monroe",
        "la monroe": "louisiana monroe",
        "ul lafayette": "louisiana",
        "louisiana lafayette": "louisiana",
        "western ky": "western kentucky",
        "san jose st": "san jose state",
        "sj state": "san jose state",
        "fresno st": "fresno state",
        "southern miss": "southern mississippi",
        "mass": "massachusetts",
        "uconn": "connecticut",
        "cal": "california",
        "penn st": "penn state",
        "southern cal": "southern california",
        "la lafayette": "louisiana",
        "nc st": "nc state",
        "ga tech": "georgia tech",
        "vt": "virginia tech",
        "uva": "virginia",
        "hawaii": "hawaii",
        "hawai'i": "hawaii",
        "hawaiʻi": "hawaii",
        "lsu": "louisiana state",
        "byu": "brigham young",
        "smu": "southern methodist",
        "tcu": "texas christian",
        "ucf": "central florida",
        "usf": "south florida",
        "utsa": "texas san antonio",
        "utep": "texas el paso",
        "unlv": "nevada las vegas",
        "umass": "massachusetts",
        "usc": "southern california",
        "fiu": "florida international",
        "fau": "florida atlantic",
        "miami fl": "miami",
        "miami fla": "miami",
        "miami oh": "miami (oh)",
    }
    alias_map.update({
        "arizona wildcats":"arizona",
        "arkansas razorbacks":"arkansas",
        "arkansas state red wolves":"arkansas state",
        "byu cougars":"brigham young",
        "boston college eagles":"boston college",
        "california golden bears":"california",
        "charlotte 49ers":"charlotte",
        "coastal carolina chanticleers":"coastal carolina",
        "colorado state rams":"colorado state",
        "east carolina pirates":"east carolina",
        "eastern michigan eagles":"eastern michigan",
        "houston cougars":"houston",
        "iowa state cyclones":"iowa state",
        "kansas state wildcats":"kansas state",
        "miami hurricanes":"miami",
        "miami (oh) redhawks":"miami (oh)",
        "mississippi state bulldogs":"mississippi state",
        "nc state wolfpack":"nc state",
        "navy midshipmen":"navy",
        "nevada wolf pack":"nevada",
        "north carolina tar heels":"north carolina",
        "oklahoma state cowboys":"oklahoma state",
        "old dominion monarchs":"old dominion",
        "san diego st aztecs":"san diego state",
        "san jose st spartans":"san jose state",
        "south carolina gamecocks":"south carolina",
        "stanford cardinal":"stanford",
        "uab blazers":"uab",
        "ucf knights":"central florida",
        "unlv rebels":"nevada las vegas",
        "usc trojans":"southern california",
        "utah utes":"utah",
        "utep miners":"texas el paso",
        "utsa roadrunners":"texas san antonio",
        "washington huskies":"washington",
        "western michigan broncos":"western michigan",
        "appalachian state mountaineers":"appalachian state",
        "virginia cavaliers":"virginia",
        "texas tech red raiders":"texas tech",
        "baylor bears":"baylor",
        "purdue boilermakers":"purdue",
        "air force falcons":"air force",
        "wyoming cowboys":"wyoming",
        "georgia bulldogs":"georgia",
        "georgia tech yellow jackets":"georgia tech",
        "boise state broncos":"boise state",
        "washington state cougars":"washington state"
    })

    # External overrides
    try:
        for _pth in (os.path.join(DATA_DIR, "team_aliases.json"), "agents/team_aliases.json", "data/team_aliases.json"):
            if os.path.exists(_pth):
                with open(_pth, "r") as _f:
                    _extra = json.load(_f) or {}
                _norm_extra = {clean(str(k)): clean(str(v)) for k, v in _extra.items() if isinstance(k, str) and isinstance(v, str)}
                alias_map.update(_norm_extra)
                break
    except Exception:
        pass

    def alias(s: str) -> str:
        cs = clean(s)
        return alias_map.get(cs, cs)

    if "home_team" not in schedule_df.columns or "away_team" not in schedule_df.columns:
        return None, detail

    schools = set(str(x).strip() for x in schedule_df["home_team"].dropna().unique()) | set(
        str(x).strip() for x in schedule_df["away_team"].dropna().unique()
    )

    norm_map: Dict[str, str] = {}
    acro_map: Dict[str, str] = {}
    token_index: List[Tuple[str, set]] = []

    for sch in schools:
        can = alias(sch)
        norm_map[can] = sch
        ac = acronym_from(sch)
        if ac:
            acro_map[ac] = sch
        token_index.append((sch, set(can.split())))

    q_raw = name
    q_norm = alias(q_raw)
    detail["q_norm"] = q_norm

    if q_norm in norm_map:
        return norm_map[q_norm], detail

    q_np = no_paren(q_norm)
    detail["q_no_paren"] = q_np
    if q_np in norm_map:
        return norm_map[q_np], detail

    q_alias = alias(q_np)
    detail["q_alias"] = q_alias
    if q_alias in norm_map:
        return norm_map[q_alias], detail

    q_acro = acronym_from(q_raw)
    if q_acro and q_acro in acro_map:
        return acro_map[q_acro], detail

    q_tokens = set(q_alias.split())
    best_team, best_score = None, 0.0
    for sch, toks in token_index:
        if not toks:
            continue
        inter = len(q_tokens & toks)
        if inter == 0:
            continue
        union = len(q_tokens | toks)
        jacc = inter / float(union) if union else 0.0
        contain = 1.0 if (q_tokens.issubset(toks) or toks.issubset(q_tokens)) else 0.0
        score = jacc + 0.25 * contain
        if score > best_score:
            best_score, best_team = score, sch

    if best_team and best_score >= 0.40:
        detail.update({"best_team": best_team, "best_score": best_score})
        return best_team, detail

    detail.update({"best_team": best_team, "best_score": best_score})
    return None, detail


def _autofix_aliases_from_unmatched(
    unmatched_json_path: str = os.path.join(DATA_DIR, "market_unmatched.json"),
    alias_json_path: str = os.path.join(DATA_DIR, "team_aliases.json"),
    min_score: float = 0.86,
) -> Dict[str, str]:
    """
    Read market_unmatched.json and auto-generate alias entries for FanDuel/raw names that had strong fuzzy matches.
    Saves/merges into data/team_aliases.json. Returns the dict of aliases added.
    """
    try:
        if not os.path.exists(unmatched_json_path):
            return {}
        with open(unmatched_json_path, "r") as f:
            payload = json.load(f) or {}
        items = payload.get("unmatched", payload) or []

        alias_map: Dict[str, str] = {}
        if os.path.exists(alias_json_path):
            try:
                with open(alias_json_path, "r") as af:
                    alias_map = json.load(af) or {}
            except Exception:
                alias_map = {}
        added: Dict[str, str] = {}

        def _norm_for_alias(s: str) -> str:
            s = (s or "").lower().strip()
            s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
            s = s.replace(" st.", " state").replace(" st ", " state ")
            import re
            s = re.sub(r"[^a-z0-9() ]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        for u in items:
            fd_h = u.get("fd_home")
            h_best = u.get("home_fuzzy") or u.get("home_best")
            h_score = float(u.get("home_fuzzy_score") or u.get("home_best_score") or 0.0)
            h_res = u.get("home_resolved")

            fd_a = u.get("fd_away")
            a_best = u.get("away_fuzzy") or u.get("away_best")
            a_score = float(u.get("away_fuzzy_score") or u.get("away_best_score") or 0.0)
            a_res = u.get("away_resolved")

            target_h = h_res or (h_best if h_score >= min_score else None)
            target_a = a_res or (a_best if a_score >= min_score else None)

            if fd_h and target_h:
                k = _norm_for_alias(fd_h)
                v = _norm_for_alias(target_h)
                if k and v and k != v and alias_map.get(k) != v:
                    alias_map[k] = v
                    added[k] = v
            if fd_a and target_a:
                k = _norm_for_alias(fd_a)
                v = _norm_for_alias(target_a)
                if k and v and k != v and alias_map.get(k) != v:
                    alias_map[k] = v
                    added[k] = v

        if added:
            os.makedirs(os.path.dirname(alias_json_path), exist_ok=True)
            with open(alias_json_path, "w") as af:
                json.dump(alias_map, af, indent=2, sort_keys=True)
            _dbg(f"autofix_aliases_from_unmatched: added {len(added)} alias entries -> {alias_json_path}")
        else:
            _dbg("autofix_aliases_from_unmatched: no strong fuzzy candidates to add")
        return added
    except Exception as e:
        print(f"[warn] autofix aliases failed: {e}", file=sys.stderr)
        return {}


def get_market_lines_fanduel_for_weeks(
    year: int, weeks: List[int], schedule_df: pd.DataFrame, cache: ApiCache
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = _odds_api_fetch_fanduel(year, weeks, cache)
    raw_count = len(raw)
    _dbg(f"get_market_lines_fanduel_for_weeks: raw games from odds api={raw_count}")
    if not raw:
        stats = {"raw": raw_count, "mapped": 0, "unmatched": 0}
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"]), stats

    # Pre-index schedule
    idx = {}
    for _, row in schedule_df.iterrows():
        a = str(row.get("away_team") or "").strip()
        h = str(row.get("home_team") or "").strip()
        if a and h:
            idx[(a, h)] = {"game_id": row.get("game_id"), "week": row.get("week")}

    # Schedule by date for constrained matching
    sched_by_date: Dict[str, Dict[str, Any]] = {}
    def _clean_local(x: str) -> str:
        s = (x or "").lower().strip()
        s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
        s = s.replace(" st.", " state").replace(" st ", " state ")
        import re
        s = re.sub(r"[^a-z0-9() ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    for _, row in schedule_df.iterrows():
        d = str(row.get("date") or "").strip()
        if not d:
            continue
        h = str(row.get("home_team") or "").strip()
        a = str(row.get("away_team") or "").strip()
        ent = sched_by_date.setdefault(d, {"pairs": set(), "home_set": set(), "away_set": set(), "teams": set()})
        ent["pairs"].add((a, h))
        ent["home_set"].add(h)
        ent["away_set"].add(a)
        ent["teams"].update([h, a])

    out_rows: List[Dict[str, Any]] = []
    unmatched_details: List[Dict[str, Any]] = []

    for g in raw:
        h_raw = g.get("home_name")
        a_raw = g.get("away_name")

        h_name, h_dbg = _resolve_names_to_schedule_with_details(schedule_df, h_raw)
        a_name, a_dbg = _resolve_names_to_schedule_with_details(schedule_df, a_raw)

        if not h_name or not a_name:
            # Constrain to same date and fuzzy within that slate
            cdate = _date_from_iso(g.get("commence_time"))
            same_date = sched_by_date.get(cdate, {}) if cdate else {}
            cand_teams = same_date.get("teams") if same_date else None
            if not cand_teams:
                cand_teams = set(str(x).strip() for x in schedule_df.get("home_team", pd.Series([], dtype=str)).dropna()) | set(
                    str(x).strip() for x in schedule_df.get("away_team", pd.Series([], dtype=str)).dropna()
                )

            def _norm(s: str) -> str:
                return _clean_local(s)

            if not h_name:
                h_best, h_score, h_qn = _best_fuzzy_match(h_raw, cand_teams, _norm)
            else:
                h_best, h_score, h_qn = h_name, 1.0, _norm(h_raw)
            if not a_name:
                a_best, a_score, a_qn = _best_fuzzy_match(a_raw, cand_teams, _norm)
            else:
                a_best, a_score, a_qn = a_name, 1.0, _norm(a_raw)

            pair_ok = False
            if h_best and a_best:
                if cdate and same_date:
                    pair_ok = (a_best, h_best) in same_date.get("pairs", set())
                if not pair_ok:
                    pair_ok = (a_best, h_best) in idx

            if h_best and a_best and pair_ok and min(h_score, a_score) >= 0.82:
                h_name, a_name = h_best, a_best
            else:
                unmatched_details.append(
                    {
                        "fd_home": h_raw,
                        "fd_away": a_raw,
                        "home_resolved": h_name,
                        "away_resolved": a_name,
                        "home_fuzzy": h_best,
                        "home_fuzzy_score": h_score,
                        "away_fuzzy": a_best,
                        "away_fuzzy_score": a_score,
                        "commence_date": cdate,
                        "reason": "unmatched-after-fuzzy" if (h_best or a_best) else "no-resolution",
                    }
                )
                continue

        if (a_name, h_name) not in idx:
            unmatched_details.append(
                {
                    "fd_home": h_raw,
                    "fd_away": a_raw,
                    "home_resolved": h_name,
                    "away_resolved": a_name,
                    "reason": "pair-not-in-index",
                }
            )
            continue

        meta = idx.get((a_name, h_name))
        if not meta:
            unmatched_details.append(
                {
                    "fd_home": h_raw,
                    "fd_away": a_raw,
                    "home_resolved": h_name,
                    "away_resolved": a_name,
                    "reason": "resolved-names-not-in-schedule-index",
                }
            )
            continue

        try:
            spread = float(g.get("point_home_book"))
        except Exception:
            unmatched_details.append(
                {
                    "fd_home": h_raw,
                    "fd_away": a_raw,
                    "home_resolved": h_name,
                    "away_resolved": a_name,
                    "reason": "invalid-point-home-book",
                }
            )
            continue

        out_rows.append({"game_id": meta.get("game_id"), "week": meta.get("week"), "home_team": h_name, "away_team": a_name, "spread": spread})

    # Write unmatched diagnostics
    if unmatched_details:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            pd.DataFrame(unmatched_details).to_csv(os.path.join(DATA_DIR, "market_unmatched.csv"), index=False)
            with open(os.path.join(DATA_DIR, "market_unmatched.json"), "w") as f:
                json.dump({"year": year, "unmatched": unmatched_details}, f, indent=2)
            # auto-add strong fuzzy matches to aliases for next run
            try:
                alias_added = _autofix_aliases_from_unmatched(
                    unmatched_json_path=os.path.join(DATA_DIR, "market_unmatched.json"),
                    alias_json_path=os.path.join(DATA_DIR, "team_aliases.json"),
                    min_score=0.86,
                )
                if alias_added:
                    _dbg(f"alias_autofix: inserted {len(alias_added)} aliases; will apply on next pass")
            except Exception as _e:
                print(f"[warn] alias autofix failed: {_e}", file=sys.stderr)
        except Exception:
            pass

    if not out_rows:
        stats = {"raw": raw_count, "mapped": 0, "unmatched": len(unmatched_details)}
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"]), stats

    df = pd.DataFrame(out_rows)
    stats = {"raw": raw_count, "mapped": len(out_rows), "unmatched": len(unmatched_details)}
    _dbg(f"get_market_lines_fanduel_for_weeks: stats={stats}")
    return df, stats


# ======================================================
# Market: choose source, write status
# ======================================================
def _cfbd_lines_to_bookstyle(df: pd.DataFrame) -> pd.DataFrame:
    """
    CFBD returns lines in various bookmaker perspectives; we want book-style 'home line' (negative ⇒ home favored).
    If CFBD line is 'away spread' (negative ⇒ away favored), convert to home line by multiplying by -1 when needed.
    This implementation assumes df has columns: game_id, home_team, away_team, spread_home if available;
    else 'spread' from favorite/underdog orientation is mapped to home side when favorite==home.
    """
    if df.empty:
        return df
    out = df.copy()
    if "spread_home" in out.columns:
        out.rename(columns={"spread_home": "spread"}, inplace=True)
        return out[["game_id", "week", "home_team", "away_team", "spread"]]
    # attempt to construct home line
    if {"favorite", "spread", "home_team", "away_team", "game_id", "week"}.issubset(out.columns):
        # if favorite == home, home_line = -abs(spread); else home_line = +abs(spread)
        try:
            fav = out["favorite"].astype(str)
            sp = pd.to_numeric(out["spread"], errors="coerce")
            home = out["home_team"].astype(str)
            out["spread"] = np.where(fav == home, -sp.abs(), sp.abs())
            return out[["game_id", "week", "home_team", "away_team", "spread"]]
        except Exception:
            pass
    # last resort: if 'spread' exists, assume already home-line
    keep = [c for c in ["game_id", "week", "home_team", "away_team", "spread"] if c in out.columns]
    return out[keep] if keep else pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])


def get_market_lines_for_current_week(
    year: int, week: int, schedule_df: pd.DataFrame, apis: CfbdClients, cache: ApiCache
) -> pd.DataFrame:
    """
    Use CFBD or FanDuel; fetch all weeks up to 'week' and return a df with *home line* ('spread', negative=home favorite).
    Records status (used/requested/fallback) in status.json.
    """
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
                # Note: CFBD lines API may require querying per week
                lines_rows: List[pd.DataFrame] = []
                for w in range(1, int(week) + 1):
                    try:
                        ls = apis.lines_api.get_lines(year=year, week=w, season_type="both")
                        # Flatten to rows
                        rows = []
                        for ln in ls or []:
                            # Each 'ln' can have multiple lines (different books)
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


# ======================================================
# Market-only debug entrypoint for CI
# ======================================================
def market_debug_entry() -> None:
    """
    Run a market-only pass that (a) updates status.json with the market
    source used, and (b) writes small debug artefacts:
      - data/market_debug.json (summary)
      - data/market_debug.csv  (resolved market lines)

    This function intentionally does **not** mutate module-level globals.
    Any environment overrides should be provided via env vars before the
    module is imported. We only read env here and compute with locals.
    """
    try:
        # Resolve season and optional week override from env.
        year_env = os.environ.get("SEASON") or os.environ.get("YEAR")
        try:
            year = int(year_env) if year_env else datetime.utcnow().year
        except Exception:
            year = datetime.utcnow().year

        week_override = os.environ.get("CURRENT_WEEK")
        week_from_env = None
        try:
            if week_override:
                week_from_env = int(week_override)
        except Exception:
            week_from_env = None

        # Build local cache & CFBD client using current env configuration.
        cache = ApiCache(root=CACHE_DIR, days_to_live=CACHE_TTL_DAYS)
        bearer = os.environ.get("CFBD_BEARER_TOKEN", "").strip()
        apis = CfbdClients(bearer_token=bearer)

        # Load schedule and determine week.
        schedule = load_schedule_for_year(year, apis, cache)
        wk_auto = discover_current_week(schedule) or (int(schedule["week"].max()) if not schedule.empty else 1)
        week = int(week_from_env or wk_auto or 1)

        # Pull market lines (this will also upsert status.json inside).
        market_df = get_market_lines_for_current_week(year, week, schedule, apis, cache)

        # Ensure data directory exists and write artefacts.
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
        except Exception:
            pass

        try:
            market_df.to_csv(os.path.join(DATA_DIR, "market_debug.csv"), index=False)
        except Exception as e:
            print(f"[warn] could not write market_debug.csv: {e}", file=sys.stderr)

        # Build a compact summary for quick inspection.
        try:
            status_used = None
            status_path = os.path.join(DATA_DIR, "status.json")
            if os.path.exists(status_path):
                try:
                    with open(status_path, "r") as f:
                        _s = json.load(f) or {}
                    status_used = _s.get("market_source_used") or _s.get("market_source")
                except Exception:
                    status_used = None

            summary = {
                "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "requested": (MARKET_SOURCE or "cfbd"),
                "used": status_used or (MARKET_SOURCE or "cfbd"),
                "rows": int(len(market_df)),
                "odds_key_present": bool(ODDS_API_KEY),
            }
            with open(os.path.join(DATA_DIR, "market_debug.json"), "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"[warn] could not write market_debug.json: {e}", file=sys.stderr)
    except Exception as e:
        # Never crash the workflow; surface as stderr only.
        print(f"[warn] market_debug_entry failed: {e}", file=sys.stderr)