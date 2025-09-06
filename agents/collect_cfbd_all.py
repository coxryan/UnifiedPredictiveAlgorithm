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

    def __post_init__(self):
        if cfbd and self.bearer_token:
            cfg = cfbd.Configuration(access_token=self.bearer_token)
            cli = cfbd.ApiClient(cfg)
            self.teams_api = cfbd.TeamsApi(cli)
            self.players_api = cfbd.PlayersApi(cli)
            self.ratings_api = cfbd.RatingsApi(cli)
            self.games_api = cfbd.GamesApi(cli)
            self.lines_api = cfbd.BettingApi(cli)


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
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([None] * len(s), index=s.index)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100.0


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


def load_schedule_for_year(year: int, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    key = f"sched:{year}"
    ok, data = cache.get(key)
    if ok:
        return pd.DataFrame(data)

    if not apis.games_api:
        df = _dummy_schedule(year)
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
                    "date": (getattr(g, "start_date", None) or "")[:10] or None,
                    "away_team": getattr(g, "away_team", None),
                    "home_team": getattr(g, "home_team", None),
                    "neutral_site": 1 if getattr(g, "neutral_site", False) else 0,
                }
            )
        df = pd.DataFrame(rows)
        cache.set(key, df.to_dict(orient="records"))
        return df
    except Exception as e:
        print(f"[warn] schedule fetch failed: {e}", file=sys.stderr)
        df = _dummy_schedule(year)
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

        rp_df["wrps_percent_0_100"] = pd.to_numeric(rp_df["wrps_overall_percent"], errors="coerce").round(1)

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
    Use CFBD Lines API; only the requested week is fetched.
    Output columns: game_id, week, home_team, away_team, spread (book-style, negative = home favorite)
    If unavailable, return empty df (collector will treat market=0 for all games).
    """
    if not apis.lines_api:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    key = f"lines:{year}:{week}"
    ok, cached = cache.get(key)
    if ok:
        return pd.DataFrame(cached)

    try:
        lines = apis.lines_api.get_lines(year=year, week=int(week))
    except Exception as e:
        print(f"[warn] market lines fetch failed for {year} w{week}: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

    rows = []
    for ln in lines or []:
        gid = getattr(ln, "id", None)
        w = getattr(ln, "week", None)
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
        rows.append({"game_id": gid, "week": w, "home_team": ht, "away_team": at, "spread": spread})

    df = pd.DataFrame(rows)
    cache.set(key, df.to_dict(orient="records"))
    return df


# =========================
# Predictions
# =========================
def _team_rating(df_teams: pd.DataFrame) -> pd.Series:
    wrps = pd.to_numeric(df_teams.get("wrps_percent_0_100"), errors="coerce")
    talent = pd.to_numeric(df_teams.get("talent_score_0_100"), errors="coerce")
    portal = pd.to_numeric(df_teams.get("portal_net_0_100"), errors="coerce")

    wrps = wrps.fillna(wrps.median() if not wrps.dropna().empty else 50.0)
    talent = talent.fillna(talent.median() if not talent.dropna().empty else 50.0)
    portal = portal.fillna(portal.median() if not portal.dropna().empty else 50.0)

    # weights are tunable
    return (0.5 * wrps + 0.3 * talent + 0.2 * portal).clip(0, 100)


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
) -> pd.DataFrame:
    teams = team_inputs_df.copy()
    teams["rating_0_100"] = _team_rating(teams)

    sched = schedule_df.copy()
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

    # Edge & Value (internal)
    sched["edge_h"] = sched["model_h"] - sched["market_h"]
    same_side = np.sign(sched["model_h"]) == np.sign(sched["expected_h"])
    sched["value_h"] = np.where(same_side, np.abs(sched["edge_h"]), 0.0)

    # Convert to book-style for UI
    sched["MODEL (H)"] = (-1.0 * sched["model_h"]).round(1)
    sched["MARKET (H)"] = (-1.0 * sched["market_h"]).round(1)
    sched["EXPECTED (H)"] = (-1.0 * sched["expected_h"]).round(1)
    # book-style edge = (book_model - book_market) = -(model_h - market_h)
    sched["EDGE"] = (-1.0 * sched["edge_h"]).round(1)
    sched["VALUE"] = (sched["value_h"]).round(1)

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
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols].sort_values(["WEEK", "DATE", "AWAY", "HOME"], kind="stable").reset_index(drop=True)
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

    # 2) SCHEDULE
    schedule = load_schedule_for_year(season, apis, cache)
    write_csv(schedule, f"{DATA_DIR}/cfb_schedule.csv")
    print(f"[live] schedule rows: {schedule.shape[0]}", flush=True)

    # 3) CURRENT WEEK + MARKET (only current week; others default to 0)
    current_week = discover_current_week(schedule) or 1
    market = get_market_lines_for_current_week(season, current_week, schedule, apis, cache)
    print(f"[live] current_week={current_week}; market rows={market.shape[0]}", flush=True)

    # 4) PREDICTIONS (book-style)
    preds = build_predictions_book_style(season, schedule, team_inputs, market, current_week)
    write_csv(preds, f"{DATA_DIR}/upa_predictions.csv")
    print(f"[live] predictions rows: {preds.shape[0]}", flush=True)

    # 5) LIVE EDGE
    live_edge = preds.sort_values("EDGE", key=lambda s: s.abs(), ascending=False).head(500)
    write_csv(live_edge, f"{DATA_DIR}/live_edge_report.csv")

    # 6) STATUS
    status = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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