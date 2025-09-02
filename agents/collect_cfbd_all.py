#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UPA-F Collector (market-aware, calibrated, gated edges)

Writes repo-level data files:
  - data/upa_team_inputs_datadriven_v0.csv
  - data/cfb_schedule.csv
  - data/upa_predictions.csv
  - data/live_edge_report.csv
  - data/diagnostics_summary.csv
  - data/status.json
"""

from __future__ import annotations
import os, sys, json, math, argparse, re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# ---- External API ----
try:
    import cfbd
except Exception:
    print("ERROR: cfbd package not available. Ensure 'cfbd' is installed.", file=sys.stderr)
    raise

# ---- Config / Env ----
BEARER = os.environ.get("BEARER_TOKEN", "").strip()
if not BEARER:
    print("ERROR: Missing BEARER_TOKEN.", file=sys.stderr)
    sys.exit(1)

YEAR = int(os.environ.get("UPA_YEAR", "2025"))
PRIOR = YEAR - 1
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# modeling knobs
HFA_GLOBAL_DEFAULT = 2.0
EDGE_MIN = 2.0
VALUE_MIN = 1.0
ROLL_WEEKS = 3
RATING_Z_CLIP = 1.8
RATING_Z_SCALE = 10.0

# ---- CFBD clients ----
cfg = cfbd.Configuration(access_token=BEARER)
client = cfbd.ApiClient(cfg)
teams_api = cfbd.TeamsApi(client)
players_api = cfbd.PlayersApi(client)
ratings_api = cfbd.RatingsApi(client)
games_api = cfbd.GamesApi(client)
bet_api = cfbd.BettingApi(client)

# ---- Helpers ----
def _to_float(x, default=np.nan):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _normalize_percent(x):
    if pd.isna(x): return None
    try:
        x = float(x)
    except Exception:
        return None
    return x * 100.0 if x <= 1.0 else x

def _scale_minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([50.0]*len(s), index=s.index)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series([50.0]*len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100.0

def _z_clip(arr: np.ndarray, clip=RATING_Z_CLIP) -> np.ndarray:
    m = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0: return np.zeros_like(arr)
    return np.clip((arr - m) / sd, -clip, clip)

def _linreg(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5: return (1.0, 0.0)
    a, b = np.polyfit(x[mask], y[mask], 1)
    return float(a), float(b)

def _week_now_from_sched(sched: pd.DataFrame) -> int:
    wk_scored = sched.loc[sched["home_points"].notna() & sched["away_points"].notna(), "week"]
    if not wk_scored.empty: return int(wk_scored.max())
    if "market_spread_home" in sched.columns and sched["market_spread_home"].notna().any():
        return int(sched.loc[sched["market_spread_home"].notna(), "week"].max())
    return int(sched["week"].max() if not sched.empty else 1)

# ---- FBS + Inputs ----
def df_fbs(year: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    fbs = teams_api.get_fbs_teams(year=year)
    rows = [{"team": t.school, "conference": getattr(t, "conference", None) or "FBS"} for t in fbs]
    df = pd.DataFrame(rows).drop_duplicates("team")
    return df, {r["team"]: r["conference"] for _, r in df.iterrows()}

def df_returning(year: int, conferences: List[str], team_conf: Dict[str,str]) -> pd.DataFrame:
    rows = []
    for conf in conferences:
        try:
            items = players_api.get_returning_production(year=year, conference=conf)
        except Exception as e:
            print(f"[warn] returning production failed for {conf}: {e}", file=sys.stderr)
            items = []
        for it in items or []:
            rows.append({
                "team": getattr(it, "team", None),
                "conference": getattr(it, "conference", None) or team_conf.get(getattr(it, "team", None), ""),
                "_overall": getattr(it, "overall", None),
                "_offense": getattr(it, "offense", None),
                "_defense": getattr(it, "defense", None),
                "_ppa_tot": getattr(it, "total_ppa", None),
                "_ppa_off": getattr(it, "total_offense_ppa", None) or (getattr(it, "total_passing_ppa", 0) + getattr(it, "total_rushing_ppa", 0)),
                "_ppa_def": getattr(it, "total_defense_ppa", None) or getattr(it, "total_defensive_ppa", None),
            })
    df = pd.DataFrame(rows).drop_duplicates("team")
    for src, out in [("_offense","wrps_offense_percent"),
                     ("_defense","wrps_defense_percent"),
                     ("_overall","wrps_overall_percent")]:
        if src in df.columns: df[out] = df[src].apply(_normalize_percent)

    need_proxy = any((c not in df.columns) or df[c].isna().all()
                     for c in ["wrps_offense_percent","wrps_defense_percent","wrps_overall_percent"])
    if need_proxy:
        for col in ["_ppa_tot","_ppa_off","_ppa_def"]:
            if col not in df.columns: df[col] = None
        if "wrps_overall_percent" not in df.columns or df["wrps_overall_percent"].isna().all():
            df["wrps_overall_percent"] = _scale_minmax(df["_ppa_tot"])
        if "wrps_offense_percent" not in df.columns or df["wrps_offense_percent"].isna().all():
            df["wrps_offense_percent"] = _scale_minmax(df["_ppa_off"])
        if "wrps_defense_percent" not in df.columns or df["wrps_defense_percent"].isna().all():
            df["wrps_defense_percent"] = _scale_minmax(df["_ppa_def"])

    df["wrps_percent_0_100"] = pd.to_numeric(df["wrps_overall_percent"], errors="coerce").round(1)
    df["conference"] = df.get("conference").fillna(df["team"].map(team_conf)).fillna("FBS")
    return df[["team","conference","wrps_offense_percent","wrps_defense_percent","wrps_overall_percent","wrps_percent_0_100"]].copy()

def df_talent(year: int) -> pd.DataFrame:
    try:
        items = teams_api.get_talent(year=year)
    except Exception as e:
        print(f"[warn] talent fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "talent_score_0_100": []})
    df = pd.DataFrame([{"team": x.team, "talent": float(getattr(x,"talent",0) or 0)} for x in items or []])
    if df.empty: return pd.DataFrame({"team": [], "talent_score_0_100": []})
    mn, mx = df["talent"].min(), df["talent"].max()
    df["talent_score_0_100"] = 50.0 if mx==mn else ((df["talent"]-mn)/(mx-mn)*100.0).round(1)
    return df[["team","talent_score_0_100"]]

def df_portal(year: int) -> pd.DataFrame:
    try:
        portal = players_api.get_transfer_portal(year=year)
    except Exception as e:
        print(f"[warn] portal fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})
    from collections import defaultdict
    in_ct = defaultdict(int); out_ct = defaultdict(int)
    in_val = defaultdict(float); out_val = defaultdict(float)
    for p in portal or []:
        to_team = getattr(p,"destination",None) or getattr(p,"to_team",None)
        from_team = getattr(p,"origin",None) or getattr(p,"from_team",None)
        rating = getattr(p,"rating",None); stars = getattr(p,"stars",None)
        try: val = float(rating) if isinstance(rating,(int,float)) else (float(stars) if isinstance(stars,(int,float)) else 1.0)
        except Exception: val = 1.0
        if to_team: in_ct[to_team]+=1; in_val[to_team]+=val
        if from_team: out_ct[from_team]+=1; out_val[from_team]+=val
    teams = set(list(in_ct.keys())+list(out_ct.keys()))
    rows = [{"team": t, "portal_net_count": in_ct[t]-out_ct[t], "portal_net_value": in_val[t]-out_val[t]} for t in teams]
    df = pd.DataFrame(rows)
    if df.empty: return pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})
    def scale(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum()==0: return pd.Series([50.0]*len(s))
        mn, mx = s.min(), s.max()
        if mx==mn: return pd.Series([50.0]*len(s))
        return (s-mn)/(mx-mn)*100.0
    df["portal_net_0_100"] = (0.5*scale(df["portal_net_count"])+0.5*scale(df["portal_net_value"])).round(1)
    return df[["team","portal_net_0_100","portal_net_count","portal_net_value"]]

def df_prev_sos(prior: int) -> pd.DataFrame:
    try:
        srs = ratings_api.get_srs(year=prior)
    except Exception as e:
        print(f"[warn] srs fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "sos_0_100": []})
    srs_map = {x.team: float(x.rating or 0) for x in srs or []}
    df = pd.DataFrame([{"team": t, "srs": v} for t,v in srs_map.items()])
    df["rank"] = df["srs"].rank(ascending=False, method="min")
    df["sos_0_100"] = ((133 - df["rank"] + 1)/133.0*100.0).round(1)
    return df[["team","sos_0_100"]]

# ---- Build team inputs ----
def build_team_inputs() -> pd.DataFrame:
    fbs_df, conf_map = df_fbs(YEAR)
    conferences = sorted(set(fbs_df["conference"].dropna()))
    rp = df_returning(YEAR, conferences, conf_map)
    talent = df_talent(YEAR)
    portal = df_portal(YEAR)
    sos = df_prev_sos(PRIOR)

    df = fbs_df.merge(rp, on=["team","conference"], how="left") \
               .merge(talent, on="team", how="left") \
               .merge(portal, on="team", how="left") \
               .merge(sos, on="team", how="left")

    for c in ["wrps_offense_percent","wrps_defense_percent","wrps_overall_percent","wrps_percent_0_100","talent_score_0_100","portal_net_0_100","sos_0_100"]:
        if c not in df.columns: df[c]=50.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(50.0)

    # Rebalanced weights (offense-heavy)
    df["team_power_0_100"] = (
        0.55*df["wrps_offense_percent"] +
        0.25*df["wrps_defense_percent"] +
        0.15*df["talent_score_0_100"] +
        0.03*df["portal_net_0_100"] +
        0.02*df["sos_0_100"]
    ).round(1)

    df["adv_score"] = df["team_power_0_100"].astype(float)
    z = _z_clip(df["team_power_0_100"].astype(float).values)
    df["team_rating"] = (z * RATING_Z_SCALE).round(2)

    df.sort_values(["conference","team"], inplace=True, ignore_index=True)
    df.to_csv(os.path.join(DATA_DIR, "upa_team_inputs_datadriven_v0.csv"), index=False)
    print(f"Wrote data/upa_team_inputs_datadriven_v0.csv with {df.shape[0]} rows.")
    return df

# ---- Robust market lines fetcher ----
def fetch_market_lines(year: int, weeks: List[int]) -> pd.DataFrame:
    """
    Extract consensus spread per game, HOME-POSITIVE convention.
    Tries: top-level .spread, nested .lines[*].spread, and formatted strings.
    Returns columns: game_id (int), market_spread_home (float)
    """
    rows: List[Dict[str, float]] = []
    for wk in weeks:
        try:
            lines = bet_api.get_lines(year=year, week=int(wk), season_type="regular")
        except Exception as e:
            print(f"[warn] betting get_lines week {wk} failed: {e}", file=sys.stderr)
            continue

        for gl in lines or []:
            gid = getattr(gl, "game_id", None) or getattr(gl, "id", None)
            if gid is None:
                continue

            # Case A: top-level numeric
            top_spread = getattr(gl, "spread", None)
            if isinstance(top_spread, (int, float)) and math.isfinite(float(top_spread)):
                rows.append({"game_id": int(gid), "spread_home": -float(top_spread)})

            # Case B/C: nested providers
            for ln in getattr(gl, "lines", []) or []:
                sp = getattr(ln, "spread", None)
                if isinstance(sp, (int, float)) and math.isfinite(float(sp)):
                    rows.append({"game_id": int(gid), "spread_home": -float(sp)})
                    continue
                # Formatted strings like "-7.5" or "Team -7.5"
                fmt = getattr(ln, "formatted_spread", None) or getattr(ln, "formattedSpread", None) or getattr(ln, "spread_display", None)
                if isinstance(fmt, str):
                    try:
                        m = re.findall(r"[-+]?\d+(?:\.\d+)?", fmt)
                        if m:
                            rows.append({"game_id": int(gid), "spread_home": -float(m[-1])})
                    except Exception:
                        pass

    if not rows:
        print("[warn] betting: no spreads parsed; market features will be empty", file=sys.stderr)
        return pd.DataFrame(columns=["game_id", "market_spread_home"])

    df = pd.DataFrame(rows)
    # provider consensus: median per game
    df = df.groupby("game_id", as_index=False)["spread_home"].median()
    df.rename(columns={"spread_home": "market_spread_home"}, inplace=True)
    print(f"INFO betting: parsed market spreads for {df.shape[0]} games")
    return df

# ---- Build schedule (FBS-only) + merge market ----
def build_schedule(team_inputs: pd.DataFrame) -> pd.DataFrame:
    fbs_set = set(team_inputs["team"])
    rows = []
    weeks_seen = set()

    for wk in range(1, 16):
        try:
            games = games_api.get_games(year=YEAR, week=wk, season_type="regular")
        except Exception as e:
            print(f"[warn] games fetch failed week {wk}: {e}", file=sys.stderr)
            continue

        for g in games or []:
            ht = getattr(g, "home_team", None); at = getattr(g, "away_team", None)
            if not ht or not at: continue
            if ht not in fbs_set or at not in fbs_set:  # FBS only
                continue
            gid = getattr(g, "id", None)
            if gid is None: continue
            rows.append({
                "game_id": int(gid),
                "week": int(wk),
                "date": str(getattr(g, "start_date", ""))[:10],
                "home_team": ht,
                "away_team": at,
                "home_points": _to_float(getattr(g, "home_points", None)),
                "away_points": _to_float(getattr(g, "away_points", None)),
                "neutral_site": 1 if getattr(g, "neutral_site", False) else 0
            })
            weeks_seen.add(int(wk))

    sched = pd.DataFrame(rows, columns=[
        "game_id","week","date","home_team","away_team",
        "home_points","away_points","neutral_site"
    ])

    if sched.empty:
        out = os.path.join(DATA_DIR,"cfb_schedule.csv")
        sched.to_csv(out, index=False)
        print(f"[warn] no schedule rows; wrote empty {out}", file=sys.stderr)
        return sched

    # market merge (robust + type-safe)
    market_df = fetch_market_lines(YEAR, sorted(weeks_seen))
    sched["game_id"] = sched["game_id"].astype(int)
    if not market_df.empty:
        market_df["game_id"] = market_df["game_id"].astype(int)
        sched = sched.merge(market_df, on="game_id", how="left")
    else:
        sched["market_spread_home"] = np.nan

    # add conferences
    conf_map = {r.team: r.conference for r in team_inputs[["team","conference"]].itertuples(index=False)}
    sched["home_conf"] = sched["home_team"].map(conf_map)
    sched["away_conf"] = sched["away_team"].map(conf_map)

    out = os.path.join(DATA_DIR,"cfb_schedule.csv")
    sched.to_csv(out, index=False)
    have_mkt = int(sched["market_spread_home"].notna().sum())
    print(f"Wrote {out} with {sched.shape[0]} rows ({have_mkt} with market).")
    return sched

# ---- Market-implied ratings & conference HFA ----
def solve_market_ratings(sched: pd.DataFrame, team_list: List[str]) -> Tuple[Dict[str,float], float, Dict[str,float]]:
    df = sched.dropna(subset=["market_spread_home"]).copy()
    if df.empty:
        return {t:0.0 for t in team_list}, HFA_GLOBAL_DEFAULT, {}

    non_neu = df[df["neutral_site"]==0]
    HFA0 = float(non_neu["market_spread_home"].mean()) if not non_neu.empty else HFA_GLOBAL_DEFAULT

    t_index = {t:i for i,t in enumerate(team_list)}
    n = len(team_list)
    rows_A, rows_y = [], []
    for r in df.itertuples(index=False):
        h, a, neu, y = r.home_team, r.away_team, int(r.neutral_site), float(r.market_spread_home)
        row = np.zeros(n)
        row[t_index[h]] = 1.0
        row[t_index[a]] = -1.0
        y_adj = y - (HFA0*(1-neu))
        rows_A.append(row); rows_y.append(y_adj)
    if not rows_A:
        return {t:0.0 for t in team_list}, HFA0, {}

    A = np.vstack(rows_A); y = np.array(rows_y)
    lam = 1.0
    AtA = A.T @ A + lam*np.eye(n)
    Aty = A.T @ y
    try:
        rvec = np.linalg.solve(AtA, Aty)
    except Exception:
        rvec = np.zeros(n)
    market_ratings = {t: float(rvec[t_index[t]]) for t in team_list}

    conf_hfa: Dict[str,float] = {}
    for conf in sorted(df["home_conf"].dropna().unique()):
        sub = df[(df["home_conf"]==conf) & (df["neutral_site"]==0)]
        if sub.empty: continue
        res = []
        for r in sub.itertuples(index=False):
            res.append(r.market_spread_home - (market_ratings.get(r.home_team,0)-market_ratings.get(r.away_team,0)))
        mean = float(np.nanmean(res)) if len(res) else HFA0
        w = min(1.0, len(sub)/30.0)
        conf_hfa[conf] = float(np.clip(w*mean + (1.0-w)*HFA0, 0.5, 3.0))

    HFA0 = float(np.clip(HFA0, 0.5, 3.0))
    return market_ratings, HFA0, conf_hfa

# ---- Predictions ----
def build_predictions(team_inputs: pd.DataFrame, sched: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    teams = list(team_inputs["team"])
    mrkt_ratings, HFA_global, conf_hfa = solve_market_ratings(sched, teams)

    now_week = _week_now_from_sched(sched)
    w_market = float(np.clip(0.4 + 0.05*(max(1,now_week)-1), 0.4, 0.7))

    base = team_inputs[["team","team_rating","adv_score","conference"]].copy()
    base["market_rating"] = base["team"].map(mrkt_ratings).fillna(0.0)
    base["blend_rating"] = ( (1-w_market)*base["team_rating"].astype(float) + w_market*base["market_rating"].astype(float) ).round(3)

    df = sched.merge(base.rename(columns={"team":"home_team","team_rating":"home_team_rating","adv_score":"home_adv_score","blend_rating":"home_blend","conference":"home_conf2"}),
                     on="home_team", how="left") \
             .merge(base.rename(columns={"team":"away_team","team_rating":"away_team_rating","adv_score":"away_adv_score","blend_rating":"away_blend","conference":"away_conf2"}),
                     on="away_team", how="left")

    def hfa_for_row(r):
        if int(r.neutral_site)==1: return 0.0
        conf = r.home_conf
        return float(conf_hfa.get(conf, HFA_global))
    df["hfa_points"] = [hfa_for_row(r) for r in df.itertuples(index=False)]

    # model (home-positive) → rolling calibration vs market (also home-positive)
    df["model_spread_home_raw"] = (df["home_blend"] + df["hfa_points"]) - df["away_blend"]
    have_mkt = df["market_spread_home"].notna()
    if have_mkt.any():
        weeks = sorted(df.loc[have_mkt,"week"].unique())
        use_weeks = set(weeks[-ROLL_WEEKS:])
        mask = have_mkt & df["week"].isin(use_weeks)
        a1,b1 = _linreg(df.loc[mask,"model_spread_home_raw"].values, df.loc[mask,"market_spread_home"].values)
        a1 = float(np.clip(a1, 0.8, 2.0))
    else:
        a1,b1 = 1.0, 0.0
    df["model_spread_home"] = a1*df["model_spread_home_raw"] + b1

    # advantage alignment (home-positive)
    df["adv_gap"] = (df["home_adv_score"] - df["away_adv_score"] + df["hfa_points"]).astype(float)
    if have_mkt.any():
        weeks = sorted(df.loc[have_mkt,"week"].unique())
        use_weeks = set(weeks[-ROLL_WEEKS:])
        mask = have_mkt & df["week"].isin(use_weeks)
        a2,b2 = _linreg(df.loc[mask,"adv_gap"].values, df.loc[mask,"market_spread_home"].values)
        a2 = float(np.clip(a2, 0.5, 3.0))
    else:
        a2,b2 = 1.0, 0.0
    df["expected_market_spread"] = a2*df["adv_gap"] + b2

    # book-style mirrors
    df["market_spread_book"] = -df["market_spread_home"]
    df["model_spread_book"] = -df["model_spread_home"]
    df["expected_market_spread_book"] = -df["expected_market_spread"]

    # edges/values (book-style)
    df["edge_points_book"] = df["model_spread_book"] - df["market_spread_book"]
    df["value_points_book"] = df["market_spread_book"] - df["expected_market_spread_book"]

    # outcomes
    df["played"] = df["home_points"].notna() & df["away_points"].notna()
    df["actual_home_margin"] = df["home_points"] - df["away_points"]
    def pick_from_book(x):
        if not np.isfinite(x): return None
        if x < 0: return "HOME"
        if x > 0: return "AWAY"
        return "PUSH"
    df["model_pick"] = df["model_spread_book"].apply(pick_from_book)
    df["expected_pick"] = df["expected_market_spread_book"].apply(pick_from_book)
    def winner_from_margin(m):
        if not np.isfinite(m): return None
        if m > 0: return "HOME"
        if m < 0: return "AWAY"
        return "PUSH"
    df["actual_winner"] = df["actual_home_margin"].apply(winner_from_margin)
    def correct(pred, actual):
        if pred is None or actual is None: return None
        if pred=="PUSH" or actual=="PUSH": return None
        return "CORRECT" if pred==actual else "INCORRECT"
    df["model_result"] = [correct(p,a) for p,a in zip(df["model_pick"], df["actual_winner"])]
    df["expected_result"] = [correct(p,a) for p,a in zip(df["expected_pick"], df["actual_winner"])]

    # edge gating
    same_sign = np.sign(df["model_spread_book"].astype(float)) == np.sign(df["expected_market_spread_book"].astype(float))
    has_mkt = df["market_spread_book"].notna()
    big_edge = df["edge_points_book"].abs() >= EDGE_MIN
    big_val  = df["value_points_book"].abs() >= VALUE_MIN
    df["qualified_edge_flag"] = (same_sign & has_mkt & big_edge & big_val).astype(int)
    df["qualified_reason"] = np.where(df["qualified_edge_flag"]==1, "agree+edge+value", "—")

    # tidy numeric precision
    num_cols = ["model_spread_home_raw","model_spread_home","market_spread_home","expected_market_spread",
                "market_spread_book","model_spread_book","expected_market_spread_book",
                "edge_points_book","value_points_book","hfa_points","home_blend","away_blend",
                "home_team_rating","away_team_rating","home_adv_score","away_adv_score","adv_gap"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    # live edge
    live = df.dropna(subset=["edge_points_book"]).copy()
    live["abs_edge"] = live["edge_points_book"].abs()
    live.sort_values(["week","abs_edge"], ascending=[True, False], inplace=True)
    live_out = live.drop(columns=["abs_edge"])
    live_out.to_csv(os.path.join(DATA_DIR,"live_edge_report.csv"), index=False)
    print("Wrote data/live_edge_report.csv")

    # predictions full
    keep_cols = [
        "game_id","week","date","home_team","away_team","home_conf","away_conf","neutral_site",
        "home_points","away_points","played","actual_home_margin","actual_winner",
        "home_team_rating","away_team_rating","home_blend","away_blend",
        "home_adv_score","away_adv_score","adv_gap","hfa_points",
        "market_spread_home","model_spread_home","expected_market_spread",
        "market_spread_book","model_spread_book","expected_market_spread_book",
        "edge_points_book","value_points_book",
        "model_pick","model_result","expected_pick","expected_result",
        "qualified_edge_flag","qualified_reason"
    ]
    for c in keep_cols:
        if c not in df.columns: df[c]=np.nan
    preds = df[keep_cols].copy()
    preds.to_csv(os.path.join(DATA_DIR,"upa_predictions.csv"), index=False)
    print("Wrote data/upa_predictions.csv")
    return preds, live_out

# ---- Diagnostics ----
def write_diagnostics(preds: pd.DataFrame):
    out_rows = []
    mask = preds["market_spread_book"].notna() & preds["model_spread_book"].notna()
    if mask.any():
        err = (preds.loc[mask,"model_spread_book"] - preds.loc[mask,"market_spread_book"]).abs()
        out_rows.append({"metric":"MAE_overall", "value": round(float(err.mean()),3)})
        bins = [0,3,7,14,99]; labels = ["0-3","3-7","7-14","14+"]
        mkt_abs = preds.loc[mask,"market_spread_book"].abs()
        binned = pd.cut(mkt_abs, bins=bins, labels=labels, include_lowest=True)
        for lab in labels:
            idx = (binned==lab)
            if idx.any():
                out_rows.append({"metric":f"MAE_{lab}", "value": round(float(err[idx].mean()),3)})
    q = preds[preds["qualified_edge_flag"]==1]
    out_rows.append({"metric":"qualified_edges", "value": int(q.shape[0])})
    q_played = q[q["played"]==True]
    if not q_played.empty:
        hit = (q_played["model_result"]=="CORRECT").mean()
        out_rows.append({"metric":"qualified_hit_rate", "value": round(float(hit),3)})
    pd.DataFrame(out_rows).to_csv(os.path.join(DATA_DIR,"diagnostics_summary.csv"), index=False)
    print("Wrote data/diagnostics_summary.csv")

# ---- Main ----
def main():
    start = datetime.now(timezone.utc)
    teams_df = build_team_inputs()
    sched_df = build_schedule(teams_df)
    preds_df, _ = build_predictions(teams_df, sched_df)
    write_diagnostics(preds_df)

    status = {
        "generated_at_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": YEAR,
        "teams": int(teams_df.shape[0]),
        "games": int(sched_df.shape[0]),
        "pred_rows": int(preds_df.shape[0]),
        "next_run_eta_utc": (start + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(os.path.join(DATA_DIR,"status.json"), "w") as f:
        json.dump(status, f, indent=2)
    print("Collector completed.")

if __name__ == "__main__":
    main()