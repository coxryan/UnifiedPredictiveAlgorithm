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
    df.to_csv(path, index=False)


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


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
# Market: current week only
# =========================
def get_market_lines_for_current_week(year: int, week: int, schedule_df: pd.DataFrame, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    """
    Use CFBD Lines API; fetch ALL weeks up to and including `week`.
    Output columns: game_id, week, home_team, away_team, spread (book-style, negative = home favorite)
    If unavailable, return empty df (collector will treat market=0 for future games without lines).
    """
    if not apis.lines_api:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"]) 

    # Determine which weeks to fetch (≤ week)
    try:
        w = int(week)
    except Exception:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"]) 

    weeks = sorted({int(x) for x in pd.to_numeric(schedule_df.get("week"), errors="coerce").dropna().astype(int) if int(x) <= w})
    all_rows = []
    for wk in weeks:
        key = f"lines:{year}:{wk}"
        ok, cached = cache.get(key)
        if ok:
            df = pd.DataFrame(cached)
        else:
            # Respect CACHE_ONLY: don't make a network call on a cache miss
            if CACHE_ONLY:
                cache.set(key, [])
                df = pd.DataFrame()
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
                df = pd.DataFrame(rows)
                cache.set(key, df.to_dict(orient="records"))
        if not df.empty:
            all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"]) 


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