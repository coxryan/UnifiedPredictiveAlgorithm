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
# =========================

def _elo_ranks(year: int, weeks: List[int], apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    """Fetch weekly Elo ratings and convert to per-week ranks and 0..100 score."""
    if not apis.ratings_api:
        return pd.DataFrame(columns=["week","team","elo_rank_1_133","elo_score_0_100"])
    out: List[Dict[str, Any]] = []
    for wk in sorted({int(w) for w in weeks if pd.notna(w)}):
        key = f"elo:{year}:{wk}"
        ok, data = cache.get(key)
        if not ok:
            try:
                items = apis.ratings_api.get_elo(year=year, week=int(wk)) or []
                data = [{"week": wk, "team": getattr(it, "team", None), "elo": float(getattr(it, "elo", 0) or 0)} for it in items]
                cache.set(key, data)
            except Exception as e:
                print(f"[warn] elo fetch failed for {year} w{wk}: {e}", file=sys.stderr)
                data = []
        out.extend(data)
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["week","team","elo_rank_1_133","elo_score_0_100"])
    df["elo"] = pd.to_numeric(df["elo"], errors="coerce")
    df["elo_rank_1_133"] = df.groupby("week")["elo"].rank(ascending=False, method="min").astype(int)
    maxr = df.groupby("week")["elo_rank_1_133"].transform("max").astype(float)
    df["elo_score_0_100"] = (1.0 - (df["elo_rank_1_133"] - 1) / maxr) * 100.0
    return df[["week","team","elo_rank_1_133","elo_score_0_100"]]


def _poll_ranks(year: int, weeks: List[int], apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    """Fetch weekly AP/Coaches poll ranks (Top 25). Returns NaN for unranked teams."""
    if not apis.rankings_api:
        return pd.DataFrame(columns=["week","team","ap_rank","coaches_rank"])
    rows: List[Dict[str, Any]] = []
    for wk in sorted({int(w) for w in weeks if pd.notna(w)}):
        key = f"polls:{year}:{wk}"
        ok, data = cache.get(key)
        if not ok:
            try:
                items = apis.rankings_api.get_rankings(year=year, week=int(wk), season_type="regular") or []
                serial: List[Dict[str, Any]] = []
                for p in items:
                    poll_name = (getattr(p, "poll", None) or getattr(p, "poll_name", None) or "").lower()
                    ranks = getattr(p, "ranks", []) or []
                    for r in ranks:
                        school = getattr(r, "school", None) or getattr(r, "team", None)
                        rank = getattr(r, "rank", None)
                        if not school or rank is None:
                            continue
                        serial.append({"poll": poll_name, "team": school, "rank": int(rank), "week": int(wk)})
                cache.set(key, serial)
                data = serial
            except Exception as e:
                print(f"[warn] rankings fetch failed for {year} w{wk}: {e}", file=sys.stderr)
                data = []
        ap_map: Dict[str, int] = {}
        co_map: Dict[str, int] = {}
        for row in data:
            poll = (row.get("poll") or "").lower()
            team = row.get("team")
            rk = row.get("rank")
            if not team or rk is None:
                continue
            if "ap" in poll:
                ap_map[team] = rk
            elif "coaches" in poll:
                co_map[team] = rk
        for t in set(list(ap_map.keys()) + list(co_map.keys())):
            rows.append({"week": int(wk), "team": t, "ap_rank": ap_map.get(t), "coaches_rank": co_map.get(t)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["week","team","ap_rank","coaches_rank"])


# =========================
# Predictions
# =========================
def _team_rating(df_teams: pd.DataFrame) -> pd.Series:
    wrps = pd.to_numeric(df_teams.get("wrps_percent_0_100"), errors="coerce")
    talent = pd.to_numeric(df_teams.get("talent_score_0_100"), errors="coerce")
    portal = pd.to_numeric(df_teams.get("portal_net_0_100"), errors="coerce")
    srs_score = pd.to_numeric(df_teams.get("srs_score_0_100"), errors="coerce")

    wrps = wrps.fillna(wrps.median() if not wrps.dropna().empty else 50.0)
    talent = talent.fillna(talent.median() if not talent.dropna().empty else 50.0)
    portal = portal.fillna(portal.median() if not portal.dropna().empty else 50.0)
    # If no SRS data yet (early season / API miss), backfill from WRPS to avoid skew
    if srs_score.dropna().empty:
        srs_score = wrps.copy()
    else:
        srs_score = srs_score.fillna(srs_score.median() if not srs_score.dropna().empty else 50.0)

    # Weights are tunable; add rank-based smoothing via SRS score
    return (0.35 * wrps + 0.25 * talent + 0.15 * portal + 0.25 * srs_score).clip(0, 100)


def _home_spread_from_ratings(home_r: float, away_r: float, neutral_flag: int) -> float:
    scale = 12.0  # 12 rating points ~ 1 TD (tune later)
    hfa = 0.0 if int(neutral_flag) == 1 else 2.0
    return (home_r - away_r) / (scale / 7.0) + hfa


def build_predictions_book_style(
    season: int,
    schedule_df: pd.DataFrame,
    team_inputs_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame],
    current_week: Optional[int],
    valid_teams: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    teams = team_inputs_df.copy()
    teams["rating_0_100"] = _team_rating(teams)

    # Normalize allowed team names and hard-filter schedule to FBS vs FBS only
    norm_ok: Optional[set] = None
    if valid_teams is not None:
        norm_ok = {str(t).strip() for t in valid_teams if str(t).strip()}

    sched = schedule_df.copy()
    if norm_ok is not None and not sched.empty:
        sched = sched.loc[
            sched["home_team"].astype(str).str.strip().isin(norm_ok)
            & sched["away_team"].astype(str).str.strip().isin(norm_ok)
        ].copy()

    for col in ["game_id", "week", "date", "home_team", "away_team", "neutral_site"]:
        if col not in sched.columns:
            sched[col] = None

    sched = sched.merge(
        teams[["team", "rating_0_100"]].rename(columns={"team": "home_team", "rating_0_100": "home_rating"}),
        on="home_team",
        how="left",
    ).merge(
        teams[["team", "rating_0_100"]].rename(columns={"team": "away_team", "rating_0_100": "away_rating"}),
        on="away_team",
        how="left",
    )
    # --- Ranks: prefer weekly Elo; fallback to season SRS ---
    weeks_list = pd.to_numeric(sched.get("week"), errors="coerce").dropna().astype(int).tolist()
    elo = _elo_ranks(season, weeks_list, apis, cache)
    if not elo.empty:
        sched = sched.merge(
            elo.rename(columns={"team": "home_team", "elo_rank_1_133": "home_rank"}),
            on=["week", "home_team"], how="left",
        ).merge(
            elo.rename(columns={"team": "away_team", "elo_rank_1_133": "away_rank"}),
            on=["week", "away_team"], how="left",
        )
    else:
        # fallback to season SRS rank in team inputs
        sched = sched.merge(
            teams[["team", "srs_rank_1_133"]].rename(columns={"team": "home_team", "srs_rank_1_133": "home_rank"}),
            on="home_team", how="left",
        ).merge(
            teams[["team", "srs_rank_1_133"]].rename(columns={"team": "away_team", "srs_rank_1_133": "away_rank"}),
            on="away_team", how="left",
        )

    sched["home_rank"] = pd.to_numeric(sched["home_rank"], errors="coerce")
    sched["away_rank"] = pd.to_numeric(sched["away_rank"], errors="coerce")
    med_rank = 66 if sched[["home_rank","away_rank"]].stack().dropna().empty else int(sched[["home_rank","away_rank"]].stack().median())
    sched["home_rank"] = sched["home_rank"].fillna(med_rank).astype(float)
    sched["away_rank"] = sched["away_rank"].fillna(med_rank).astype(float)
    sched["rank_delta"] = (sched["away_rank"] - sched["home_rank"]).astype(float)

    # (Optional) Poll ranks merged for completeness (not used by UI yet)
    polls = _poll_ranks(season, weeks_list, apis, cache)
    if not polls.empty:
        sched = sched.merge(
            polls.rename(columns={"team": "home_team", "ap_rank": "home_ap_rank", "coaches_rank": "home_coaches_rank"}),
            on=["week", "home_team"], how="left",
        ).merge(
            polls.rename(columns={"team": "away_team", "ap_rank": "away_ap_rank", "coaches_rank": "away_coaches_rank"}),
            on=["week", "away_team"], how="left",
        )

    sched["home_rating"] = sched["home_rating"].fillna(50.0)
    sched["away_rating"] = sched["away_rating"].fillna(50.0)
    sched["neutral_site"] = pd.to_numeric(sched["neutral_site"], errors="coerce").fillna(0).astype(int)

    # Internal home-positive model spread
    sched["model_h"] = [
        _home_spread_from_ratings(h, a, n) for h, a, n in zip(sched["home_rating"], sched["away_rating"], sched["neutral_site"])
    ]

    # Market: current week only; others = 0 internally
    sched["market_h"] = 0.0
    if market_df is not None and not market_df.empty and current_week is not None:
        m = market_df.copy()
        # market_df is book-style (negative favors home). Convert to internal home-positive.
        m["market_h_internal"] = -1.0 * pd.to_numeric(m["spread"], errors="coerce")
        m = m.loc[pd.to_numeric(m["week"], errors="coerce").fillna(-1).astype(int) == int(current_week)]
        sched = sched.merge(
            m[["game_id", "market_h_internal"]], on="game_id", how="left", validate="one_to_one"
        )
        sched["market_h"] = sched["market_h_internal"].fillna(0.0)
        sched.drop(columns=["market_h_internal"], inplace=True, errors="ignore")

    # Expected = market (hook for smoothing later)
    sched["expected_h"] = sched["market_h"]

    # --- Calibrate model using market & rank delta (linear least squares) ---
    sched["model_h_base"] = sched["model_h"].astype(float)
    alpha, beta, gamma = 0.0, 1.0, 0.0
    cal = sched.loc[
        pd.notna(sched["market_h"]) & pd.notna(sched["model_h_base"]) & pd.notna(sched["rank_delta"])
    , ["market_h", "model_h_base", "rank_delta"]].copy()
    if cal.shape[0] >= 8:
        try:
            X = np.column_stack([
                np.ones(len(cal)),
                cal["model_h_base"].to_numpy(float),
                cal["rank_delta"].to_numpy(float),
            ])
            y = cal["market_h"].to_numpy(float)
            theta, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta, gamma = float(theta[0]), float(theta[1]), float(theta[2])
        except Exception:
            # Fallback: only adjust on rank_delta → residual regression
            denom = float(np.maximum(1e-6, np.var(cal["rank_delta"].to_numpy(float))))
            resid = (cal["market_h"].to_numpy(float) - cal["model_h_base"].to_numpy(float))
            num = float(np.cov(cal["rank_delta"].to_numpy(float), resid, bias=True)[0, 1])
            gamma = num / denom
            alpha, beta = 0.0, 1.0
    else:
        # Default gentle rank influence when little data: ~1 point per 100 rank places
        alpha, beta, gamma = 0.0, 1.0, 0.01

    # Bound coefficients to keep adjustments reasonable
    beta = float(np.clip(beta, 0.6, 1.4))
    gamma = float(np.clip(gamma, -0.02, 0.02))  # ≤2 pts per 100-rank delta
    alpha = float(np.clip(alpha, -3.0, 3.0))

    # Adjusted model (home-positive)
    sched["model_h_adj"] = alpha + beta * sched["model_h_base"].astype(float) + gamma * sched["rank_delta"].astype(float)
    print(f"[live] rank-calibration: alpha={alpha:.3f} beta={beta:.3f} gamma={gamma:.3f} n={cal.shape[0]}", flush=True)

    # Edge & Value using adjusted model
    sched["edge_h"] = sched["model_h_adj"].astype(float) - sched["market_h"].astype(float)
    same_side = np.sign(sched["model_h_adj"]) == np.sign(sched["expected_h"])
    sched["value_h"] = np.where(same_side, np.abs(sched["edge_h"]), 0.0)

    # Convert to book-style for UI (also keep base for debugging)
    sched["MODEL_BASE (H)"] = (-1.0 * sched["model_h_base"]).round(1)
    sched["MODEL (H)"] = (-1.0 * sched["model_h_adj"]).round(1)
    sched["MARKET (H)"] = (-1.0 * sched["market_h"]).round(1)
    sched["EXPECTED (H)"] = (-1.0 * sched["expected_h"]).round(1)
    # book-style edge = (book_model - book_market) = -(model_h - market_h)
    sched["EDGE"] = (-1.0 * sched["edge_h"]).round(1)
    sched["VALUE"] = (sched["value_h"]).round(1)

    # --- Outcomes & correctness (for past weeks) ---
    # played if we have both scores
    sched["home_points"] = pd.to_numeric(sched.get("home_points"), errors="coerce")
    sched["away_points"] = pd.to_numeric(sched.get("away_points"), errors="coerce")
    sched["played_flag"] = np.where(pd.notna(sched["home_points"]) & pd.notna(sched["away_points"]), 1, 0)

    # Actual ATS from home perspective using internal lines
    # market_h (internal) already computed; margin_home = home_points - away_points
    margin_home = (sched["home_points"] - sched["away_points"]).astype(float)
    ats_score = margin_home - sched["market_h"].astype(float)
    # model pick from adjusted model vs market: edge_h > 0 ⇒ value = AWAY, else HOME
    model_pick = np.where((sched["model_h_adj"].astype(float) - sched["market_h"].astype(float)) > 0, "AWAY", "HOME")

    result = np.where(sched["played_flag"].eq(1),
                      np.where(ats_score > 0, "HOME",
                               np.where(ats_score < 0, "AWAY", "PUSH")), "")
    sched["model_result"] = np.where(sched["played_flag"].eq(1),
                                      np.where(result == "PUSH", "PUSH",
                                               np.where(result == model_pick, "CORRECT", "INCORRECT")), "")

    out = sched.rename(
        columns={
            "week": "WEEK",
            "date": "DATE",
            "away_team": "AWAY",
            "home_team": "HOME",
            "neutral_site": "NEUTRAL",
        }
    )
    out["NEUTRAL"] = np.where(out["NEUTRAL"].astype(int) == 1, "Y", "—")
    # Carry ranks to output
    out["HOME_RANK"] = pd.to_numeric(sched.get("home_rank"), errors="coerce")
    out["AWAY_RANK"] = pd.to_numeric(sched.get("away_rank"), errors="coerce")
    out["HOME_POINTS"] = sched.get("home_points")
    out["AWAY_POINTS"] = sched.get("away_points")

    cols = [
        "WEEK",
        "DATE",
        "AWAY",
        "HOME",
        "NEUTRAL",
        "MODEL (H)",
        "MARKET (H)",
        "EXPECTED (H)",
        "EDGE",
        "VALUE",
        "game_id",
        "HOME_RANK",
        "AWAY_RANK",
        "HOME_POINTS",
        "AWAY_POINTS",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan

    # --- UI-compatible columns (book-style, home-negative) ---
    # Map to lower-case names the React app expects
    out["week"] = out["WEEK"]
    out["date"] = out["DATE"]
    out["home_team"] = out["HOME"]
    out["away_team"] = out["AWAY"]
    out["home_rank"] = out["HOME_RANK"]
    out["away_rank"] = out["AWAY_RANK"]
    out["neutral_site"] = np.where(out["NEUTRAL"].astype(str).str.upper().eq("Y"), "1", "0")
    out["model_spread_book"] = out["MODEL (H)"]
    out["model_base_spread_book"] = out["MODEL_BASE (H)"]
    out["market_spread_book"] = out["MARKET (H)"]
    out["expected_market_spread_book"] = out["EXPECTED (H)"]
    out["edge_points_book"] = out["EDGE"]
    out["value_points_book"] = out["VALUE"]
    out["home_points"] = out["HOME_POINTS"]
    out["away_points"] = out["AWAY_POINTS"]
    out["played"] = np.where(sched["played_flag"].eq(1), "1", "")

    # Qualification flag to match UI thresholds (EDGE_MIN=2.0, VALUE_MIN=1.0)
    EDGE_MIN_T = 2.0
    VALUE_MIN_T = 1.0
    edge_abs = pd.to_numeric(out["EDGE"], errors="coerce").abs()
    val_abs = pd.to_numeric(out["VALUE"], errors="coerce").abs()
    out["qualified_edge_flag"] = np.where((edge_abs >= EDGE_MIN_T) & (val_abs >= VALUE_MIN_T), "1", "0")

    # Optional placeholders used by Team & Backtest tabs
    if "played" not in out.columns:
        out["played"] = ""
    if "model_result" not in out.columns:
        out["model_result"] = ""

    out = out[cols + [
        "week","date","home_team","away_team","home_rank","away_rank","neutral_site",
        "model_spread_book","model_base_spread_book","market_spread_book","expected_market_spread_book",
        "edge_points_book","value_points_book","qualified_edge_flag","played","model_result",
        "home_points","away_points"
    ]].sort_values(["WEEK", "DATE", "AWAY", "HOME"], kind="stable").reset_index(drop=True)
    return out


# =========================
# (Optional) Backtest stub
# =========================
def run_backtest_stub(year: int, out_dir: str = DATA_DIR):
    # Minimal placeholder. Keep your prior backtest if you prefer.
    df = pd.DataFrame(
        [
            {"week": 1, "wins": 10, "losses": 6, "push": 0, "pct": round(10 / 16, 3)},
            {"week": 2, "wins": 9, "losses": 7, "push": 0, "pct": round(9 / 16, 3)},
        ]
    )
    write_csv(df, os.path.join(out_dir, f"backtest_summary_{year}.csv"))


# =========================
# Google Sheets (optional)
# =========================
def maybe_push_sheets(df: pd.DataFrame, sheet_title: str, sheet_id_env: str = "SHEET_ID"):
    if not ENABLE_SHEETS:
        return
    SHEET_ID = os.environ.get(sheet_id_env, "").strip()
    SA = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not (SHEET_ID and SA and os.path.exists(SA)):
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(SA, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        try:
            ws = sh.worksheet(sheet_title)
        except Exception:
            ws = sh.add_worksheet(title=sheet_title, rows=str(len(df) + 10), cols=str(len(df.columns) + 5))
        ws.clear()
        ws.update([df.columns.tolist()] + df.fillna("").values.tolist())
        print(f"[sheets] updated: {sheet_title}", flush=True)
    except Exception as e:
        print(f"[warn] Sheets update skipped/failed: {e}", file=sys.stderr)


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--backtest", type=int, required=False)
    args = parser.parse_args()

    season = int(args.year)
    backtest_year = int(args.backtest) if args.backtest else None

    os.makedirs(DATA_DIR, exist_ok=True)

    cache = ApiCache(root=CACHE_DIR, days_to_live=CACHE_TTL_DAYS)
    apis = CfbdClients(bearer_token=os.environ.get("BEARER_TOKEN", "").strip())

    # 1) TEAM INPUTS
    team_inputs = build_team_inputs_datadriven(season, apis, cache)
    write_csv(team_inputs, f"{DATA_DIR}/upa_team_inputs_datadriven_v0.csv")
    maybe_push_sheets(team_inputs, "Team Inputs")
    print(f"[live] team inputs for {season}: {team_inputs.shape}", flush=True)

    # Build FBS set from team_inputs for schedule filter
    fbs_names = set(pd.Series(team_inputs.get("team", [])).dropna().astype(str))

    # 2) SCHEDULE
    schedule = load_schedule_for_year(season, apis, cache, fbs_set=fbs_names)
    write_csv(schedule, f"{DATA_DIR}/cfb_schedule.csv")
    print(f"[live] schedule rows: {schedule.shape[0]}", flush=True)
    # Sanity: confirm FBS-only filtering
    raw_sched_rows = int(schedule.shape[0])

    # 3) CURRENT WEEK + MARKET (only current week; others default to 0)
    current_week = discover_current_week(schedule) or 1
    market = get_market_lines_for_current_week(season, current_week, schedule, apis, cache)
    print(f"[live] current_week={current_week}; market rows={market.shape[0]}", flush=True)

    # 4) PREDICTIONS (book-style)
    preds = build_predictions_book_style(season, schedule, team_inputs, market, current_week, valid_teams=fbs_names)
    print(f"[live] predictions rows (FBS-only): {preds.shape[0]} from schedule={raw_sched_rows}", flush=True)
    write_csv(preds, f"{DATA_DIR}/upa_predictions.csv")

    # 5) LIVE EDGE
    live_edge = preds.sort_values("EDGE", key=lambda s: s.abs(), ascending=False).head(500)
    write_csv(live_edge, f"{DATA_DIR}/live_edge_report.csv")

    # 6) STATUS
    now_utc = datetime.now(timezone.utc)
    status = {
        "generated_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "next_run_eta_utc": (now_utc + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": season,
        "current_week": current_week,
        "teams": int(team_inputs.shape[0]),
        "games": int(schedule.shape[0]),
        "pred_rows": int(preds.shape[0]),
        "note": "Market lines only fetched for current week; other weeks default to 0.",
    }
    with open(f"{DATA_DIR}/status.json", "w") as f:
        json.dump(status, f, indent=2)

    # 7) OPTIONAL BACKTEST
    if backtest_year:
        run_backtest_stub(backtest_year)

    print("[done] collectors completed.", flush=True)


if __name__ == "__main__":
    main()