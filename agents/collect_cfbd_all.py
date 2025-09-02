#!/usr/bin/env python3
import os
import sys
import json
import math
import argparse
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd

# -----------------------------
# Imports / CFBD client setup
# -----------------------------
try:
    import cfbd
except Exception as e:
    print("ERROR: cfbd package not available. Ensure Actions installs 'cfbd'.", file=sys.stderr)
    raise

BEARER = os.environ.get("BEARER_TOKEN", "").strip()
if not BEARER:
    print("ERROR: Missing BEARER_TOKEN secret for CollegeFootballData API.", file=sys.stderr)
    sys.exit(1)

configuration = cfbd.Configuration(access_token=BEARER)
api_client = cfbd.ApiClient(configuration)

teams_api   = cfbd.TeamsApi(api_client)
players_api = cfbd.PlayersApi(api_client)
ratings_api = cfbd.RatingsApi(api_client)
games_api   = cfbd.GamesApi(api_client)
bet_api     = cfbd.BettingApi(api_client)  # market spreads

# -----------------------------
# CLI / Env params
# -----------------------------
parser = argparse.ArgumentParser(description="UPA-F data collector")
parser.add_argument("--year", type=int, default=int(os.environ.get("UPA_YEAR", "2025")))
parser.add_argument("--backtest", type=int, default=int(os.environ.get("UPA_BACKTEST_YEAR", "0")))
args = parser.parse_args()

YEAR = int(args.year)
BACKTEST_YEAR = int(args.backtest) if int(args.backtest) and int(args.backtest) != YEAR else 0

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# thresholds (used in UI and here for "qualified" flag parity)
EDGE_MIN = float(os.environ.get("UPA_EDGE_MIN", "2"))
VALUE_MIN = float(os.environ.get("UPA_VALUE_MIN", "1"))

# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def _normalize_pct(x):
    """Coerce 0..1 to 0..100 else pass-through."""
    x = _safe_float(x)
    if not math.isfinite(x):
        return float("nan")
    return x * 100.0 if x <= 1.0 else x

def _scale_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([float("nan")] * len(s), index=s.index)
    mn, mx = s.min(), s.max()
    if not math.isfinite(mn) or not math.isfinite(mx) or mx == mn:
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100.0

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _iso_date(dt_str):
    if not dt_str:
        return ""
    # cfbd start_date often ISO; keep as is
    try:
        return str(dt_str)[:10]
    except Exception:
        return str(dt_str)

# -----------------------------
# Module: Team Inputs (WRPS/Talent/Portal/SOS)
# -----------------------------
EXPECTED_RETURNING_COLS = [
    "team","conference",
    "wrps_offense_percent","wrps_defense_percent","wrps_overall_percent","wrps_percent_0_100"
]

def df_returning_production(year: int) -> pd.DataFrame:
    fbs = teams_api.get_fbs_teams(year=year)
    team_conf = {t.school: (t.conference or "FBS") for t in fbs}
    conferences = sorted({t.conference for t in fbs if t.conference})

    rows = []
    for conf in conferences:
        try:
            items = players_api.get_returning_production(year=year, conference=conf)
        except Exception as e:
            print(f"[warn] returning production fetch failed for {conf}: {e}", file=sys.stderr)
            items = []
        for it in items or []:
            rows.append({
                "team": getattr(it, "team", None),
                "conference": getattr(it, "conference", None) or team_conf.get(getattr(it,"team",None), "FBS"),
                "_overall": getattr(it, "overall", None),
                "_offense": getattr(it, "offense", None),
                "_defense": getattr(it, "defense", None),
                "_ppa_tot": getattr(it, "total_ppa", None),
                "_ppa_off": getattr(it, "total_offense_ppa", None) or (getattr(it, "total_passing_ppa", 0) + getattr(it, "total_rushing_ppa", 0)),
                "_ppa_def": getattr(it, "total_defense_ppa", None) or getattr(it, "total_defensive_ppa", None),
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])

    # normalize direct %
    df["wrps_offense_percent"] = df["_offense"].apply(_normalize_pct)
    df["wrps_defense_percent"] = df["_defense"].apply(_normalize_pct)
    df["wrps_overall_percent"] = df["_overall"].apply(_normalize_pct)

    # proxy with PPA if missing
    need_proxy = (
        df["wrps_overall_percent"].isna().all() or
        df["wrps_offense_percent"].isna().all() or
        df["wrps_defense_percent"].isna().all()
    )
    if need_proxy:
        if "wrps_overall_percent" not in df or df["wrps_overall_percent"].isna().all():
            df["wrps_overall_percent"] = _scale_0_100(df["_ppa_tot"])
        if "wrps_offense_percent" not in df or df["wrps_offense_percent"].isna().all():
            df["wrps_offense_percent"] = _scale_0_100(df["_ppa_off"])
        if "wrps_defense_percent" not in df or df["wrps_defense_percent"].isna().all():
            df["wrps_defense_percent"] = _scale_0_100(df["_ppa_def"])

    df["wrps_percent_0_100"] = pd.to_numeric(df["wrps_overall_percent"], errors="coerce").round(1)
    for c in ["team","conference"] + EXPECTED_RETURNING_COLS[2:]:
        if c not in df.columns:
            df[c] = None

    # fill conference
    df["conference"] = df["team"].map(team_conf).fillna(df["conference"]).fillna("FBS")
    return df[EXPECTED_RETURNING_COLS].copy()

def df_talent(year: int) -> pd.DataFrame:
    try:
        items = teams_api.get_talent(year=year)
    except Exception as e:
        print(f"[warn] talent fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "talent_score_0_100": []})
    rows = [{"team": x.team, "talent": float(getattr(x, "talent", 0) or 0)} for x in items or []]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"team": [], "talent_score_0_100": []})
    mn, mx = df["talent"].min(), df["talent"].max()
    df["talent_score_0_100"] = 50.0 if mx == mn else ((df["talent"] - mn) / (mx - mn) * 100.0).round(1)
    return df[["team","talent_score_0_100"]]

def df_prev_sos_rank(prior_year: int) -> pd.DataFrame:
    try:
        srs = ratings_api.get_srs(year=prior_year)
    except Exception as e:
        print(f"[warn] srs fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})
    srs_map = {x.team: float(x.rating or 0) for x in srs}

    try:
        games = games_api.get_games(year=prior_year, season_type="both")
    except Exception as e:
        print(f"[warn] games fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})

    opps = defaultdict(list)
    for g in games:
        ht, at = getattr(g, "home_team", None), getattr(g, "away_team", None)
        if not ht or not at: continue
        if ht in srs_map and at in srs_map:
            opps[ht].append(srs_map[at])
            opps[at].append(srs_map[ht])

    rows = [{"team": t, "sos_value": sum(v)/len(v)} for t, v in opps.items() if v]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})
    df["prev_season_sos_rank_1_133"] = df["sos_value"].rank(ascending=False, method="min").astype(int)
    return df[["team","prev_season_sos_rank_1_133"]]

def df_transfer_portal(year: int) -> pd.DataFrame:
    try:
        portal = players_api.get_transfer_portal(year=year)
    except Exception as e:
        print(f"[warn] portal fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})

    incoming = defaultdict(int); outgoing = defaultdict(int)
    rating_in = defaultdict(float); rating_out = defaultdict(float)

    for p in portal:
        to_team = getattr(p, "destination", None) or getattr(p, "to_team", None)
        from_team = getattr(p, "origin", None) or getattr(p, "from_team", None)
        rating = getattr(p, "rating", None); stars = getattr(p, "stars", None)
        try:
            val = float(rating) if isinstance(rating, (int,float)) else (float(stars) if isinstance(stars, (int,float)) else 1.0)
        except Exception:
            val = 1.0
        if to_team:
            incoming[to_team] += 1; rating_in[to_team] += val
        if from_team:
            outgoing[from_team] += 1; rating_out[from_team] += val

    teams = set(list(incoming.keys()) + list(outgoing.keys()))
    rows = []
    for t in teams:
        cnt_net = incoming[t] - outgoing[t]
        val_net = rating_in[t] - rating_out[t]
        rows.append({"team": t, "portal_net_count": cnt_net, "portal_net_value": val_net})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})

    df["portal_net_0_100"] = (0.5*_scale_0_100(df["portal_net_count"]) + 0.5*_scale_0_100(df["portal_net_value"])).round(1)
    return df[["team","portal_net_0_100","portal_net_count","portal_net_value"]]

def build_team_inputs(year: int) -> pd.DataFrame:
    rp = df_returning_production(year)
    talent = df_talent(year)
    sos = df_prev_sos_rank(year-1)
    portal = df_transfer_portal(year)

    df = rp.merge(talent, on="team", how="left") \
           .merge(sos, on="team", how="left") \
           .merge(portal, on="team", how="left")

    if "conference" in df.columns:
        df.sort_values(["conference","team"], inplace=True, ignore_index=True)
    else:
        df.sort_values(["team"], inplace=True, ignore_index=True)
    return df

# -----------------------------
# Module: Schedule + Market
# -----------------------------
def _fbs_vs_fbs(g) -> bool:
    hc = getattr(g, "home_conference", None)
    ac = getattr(g, "away_conference", None)
    # filter out FCS games; cfbd returns None for many FCS confs
    return bool(hc) and bool(ac)

def _line_to_home_perspective(line_obj, home_team: str, away_team: str):
    """
    Try to extract a home-perspective spread from a cfbd line object.
    Return float('nan') if not found.
    """
    # cfbd models vary by season/provider; try common attributes:
    # - home_spread / away_spread
    # - spread (often away-spread; we need to infer side)
    # - formattedSpread like "Home -7.5"
    cand = []

    # explicit home
    for attr in ("home_spread", "homeSpread", "home_spread_open", "home_spread_close"):
        v = getattr(line_obj, attr, None)
        if v is not None:
            cand.append(("home", _safe_float(v)))

    # explicit away
    for attr in ("away_spread", "awaySpread", "away_spread_open", "away_spread_close"):
        v = getattr(line_obj, attr, None)
        if v is not None:
            # convert to home perspective: home = -(away)
            cand.append(("away->home", -_safe_float(v)))

    # generic spread (assume away-perspective by CFBD default; flip to home)
    v = getattr(line_obj, "spread", None)
    if v is not None:
        cand.append(("generic->home", -_safe_float(v)))

    # formattedSpread e.g., "USC -7.5"
    fs = getattr(line_obj, "formattedSpread", None) or getattr(line_obj, "formatted_spread", None)
    if isinstance(fs, str) and fs.strip():
        try:
            parts = fs.replace("–", "-").split()
            # if first token matches home or away team name, sign belongs to that side
            if len(parts) >= 2:
                sign = float(parts[1])
                first = parts[0]
                if home_team and first.lower().startswith(home_team.lower()[:4]):
                    cand.append(("fmt home", sign if sign < 0 else -abs(sign)))  # ensure negative means home favorite
                elif away_team and first.lower().startswith(away_team.lower()[:4]):
                    cand.append(("fmt away->home", -sign if sign > 0 else abs(sign)))
        except Exception:
            pass

    # choose median of available candidates (robust to outliers/open/close)
    values = [v for _, v in cand if math.isfinite(v)]
    if not values:
        return float("nan")
    values.sort()
    mid = values[len(values)//2]
    return mid

def build_schedule_with_market(year: int) -> pd.DataFrame:
    """FBS vs FBS only, combine schedule with a consensus-ish home spread."""
    games = games_api.get_games(year=year, season_type="both")
    rows = []
    for g in games:
        if not _fbs_vs_fbs(g):  # skip FCS
            continue
        ht, at = getattr(g,"home_team", ""), getattr(g,"away_team","")
        wk = getattr(g, "week", None)
        date = _iso_date(getattr(g, "start_date", None) or getattr(g, "start_time", None))
        neutral = getattr(g, "neutral_site", False)
        # attach market (median across providers for that game+week)
        market_home = float("nan")
        try:
            # Betting API lines for the specific week & teams is cheaper than all-season
            lines = bet_api.get_lines(year=year, week=wk, team=ht) or []
            # filter to this matchup (home team present)
            maybe = []
            for ln in lines:
                # each ln has 'lines' list with provider snapshots
                for snap in getattr(ln, "lines", []) or []:
                    val = _line_to_home_perspective(snap, ht, at)
                    if math.isfinite(val):
                        maybe.append(val)
            if maybe:
                maybe.sort()
                market_home = maybe[len(maybe)//2]
        except Exception as e:
            print(f"[warn] lines fetch failed wk{wk} {at}@{ht}: {e}", file=sys.stderr)

        rows.append({
            "game_id": getattr(g,"id", None),
            "week": wk,
            "date": date,
            "away_team": at,
            "home_team": ht,
            "neutral_site": "1" if bool(neutral) else "0",
            "market_spread_book": round(market_home, 1) if math.isfinite(market_home) else ""
        })
    df = pd.DataFrame(rows).sort_values(["week","date","away_team","home_team"], ignore_index=True)
    return df

# -----------------------------
# Module: Model / Predictions
# -----------------------------
def _team_advantage_score(team_row: pd.Series) -> float:
    """
    Simple composite: WRPS overall + Talent + Portal (all 0..100),
    with light weighting. You can refine weights later.
    """
    w = _safe_float(team_row.get("wrps_percent_0_100"))
    t = _safe_float(team_row.get("talent_score_0_100"))
    p = _safe_float(team_row.get("portal_net_0_100"))
    # defaults to 0 if NaN
    w = 0.0 if not math.isfinite(w) else w
    t = 0.0 if not math.isfinite(t) else t
    p = 0.0 if not math.isfinite(p) else p
    # weights
    return 0.5*w + 0.35*t + 0.15*p

def _home_field(year: int, neutral_flag: str) -> float:
    # neutral-site ~ 0; else modest HFA
    if str(neutral_flag) == "1":
        return 0.0
    # tuneable: could be conf-specific in the future
    return 2.2

def _expected_market_from_model(model_home: float, market_home: float) -> float:
    """
    Calibrate market expectation from model with a clamp, to avoid runaway value
    when market and model are far apart.
    """
    if not math.isfinite(model_home) or not math.isfinite(market_home):
        return float("nan")
    delta = model_home - market_home  # same as edge
    correction = max(-3.0, min(3.0, delta))  # clamp to +/-3 points
    return model_home - correction  # expected market home line

def _value_side_from_edge(edge: float, home_team: str, away_team: str) -> str:
    # edge = model_home − market_home
    if not math.isfinite(edge) or abs(edge) < 1e-9:
        return ""
    # edge > 0 → market too heavy on HOME → value = AWAY
    return f"{away_team} (away)" if edge > 0 else f"{home_team} (home)"

def build_predictions_for_year(year: int, team_inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with: week,date,home_team,away_team,neutral_site,
    model_spread_book, market_spread_book, expected_market_spread_book,
    edge_points_book, value_points_book, qualified_edge_flag
    """
    sched = build_schedule_with_market(year)

    # join team inputs to compute team composite advantage
    tmap = team_inputs.set_index("team", drop=False)

    def team_rating(name: str) -> float:
        if name not in tmap.index: return 50.0
        return float(_team_advantage_score(tmap.loc[name]))

    model_vals = []
    for _, r in sched.iterrows():
        home = r["home_team"]; away = r["away_team"]
        home_rt = team_rating(home)
        away_rt = team_rating(away)
        # spread home-perspective: negative means home favorite
        # map advantage (0..100) to spread by scaling around mid=50
        base_diff_pts = (home_rt - away_rt) * 0.15  # 15% of 100-pt scale => ~±15 ceiling
        hfa = _home_field(year, r["neutral_site"])
        model_home = round((-(away_rt - home_rt) * 0.15) - hfa, 1)  # equivalent, explicit sign
        # (For clarity, keep using computed base_diff_pts + HFA)
        model_home = round(base_diff_pts - hfa, 1)

        market_home = _safe_float(r.get("market_spread_book"))
        exp_home = _expected_market_from_model(model_home, market_home) if math.isfinite(market_home) else float("nan")
        edge = model_home - market_home if math.isfinite(market_home) else float("nan")
        value = market_home - exp_home if (math.isfinite(market_home) and math.isfinite(exp_home)) else float("nan")

        qual = ""
        if math.isfinite(edge) and math.isfinite(value):
            if abs(edge) >= EDGE_MIN and abs(value) >= VALUE_MIN:
                # require side agreement between model and expected (both same sign)
                if math.copysign(1, model_home) == math.copysign(1, (exp_home if math.isfinite(exp_home) else model_home)):
                    qual = "1"

        model_vals.append({
            **r.to_dict(),
            "model_spread_book": round(model_home, 1),
            "expected_market_spread_book": round(exp_home, 1) if math.isfinite(exp_home) else "",
            "edge_points_book": round(edge, 1) if math.isfinite(edge) else "",
            "value_points_book": round(value, 1) if math.isfinite(value) else "",
            "qualified_edge_flag": qual
        })

    df = pd.DataFrame(model_vals)
    # Sorting
    df.sort_values(["week","date","away_team","home_team"], inplace=True, ignore_index=True)
    return df

# -----------------------------
# Module: Backtest (value-side grading)  ✅ FIXED
# -----------------------------
def bet_result_value_side(home_points: float, away_points: float, market_home: float, model_home: float):
    """
    Grades result for the *value side* derived from edge = model - market (home perspective).
    Returns (result_str, side_str) where side_str is 'HOME' or 'AWAY'.
    """
    if not (math.isfinite(_safe_float(home_points)) and math.isfinite(_safe_float(away_points))):
        return "", ""
    if not (math.isfinite(_safe_float(market_home)) and math.isfinite(_safe_float(model_home))):
        return "", ""

    home_points = float(home_points)
    away_points = float(away_points)
    market_home = float(market_home)
    model_home = float(model_home)

    edge = model_home - market_home

    if abs(edge) < 1e-9:
        return "", ""

    if edge > 0:
        # value = AWAY; use away + (-market_home)
        spread_for_away = -market_home
        delta = (away_points - home_points) + spread_for_away
        side = "AWAY"
    else:
        # value = HOME; use home + (market_home)
        delta = (home_points - away_points) + market_home
        side = "HOME"

    if abs(delta) < 1e-9:
        return "PUSH", side
    return ("CORRECT" if delta > 0 else "INCORRECT"), side

def run_backtest(year_bt: int, team_inputs_bt: pd.DataFrame):
    out_dir = _ensure_dir(os.path.join(DATA_DIR, str(year_bt)))

    # Build predictions for backtest season
    preds = build_predictions_for_year(year_bt, team_inputs_bt).copy()

    # Pull final scores & merge
    games = games_api.get_games(year=year_bt, season_type="both")
    finals = []
    for g in games:
        if not _fbs_vs_fbs(g):  # ignore FCS
            continue
        hp = getattr(g, "home_points", None)
        ap = getattr(g, "away_points", None)
        if hp is None or ap is None:
            continue
        finals.append({
            "week": getattr(g, "week", None),
            "date": _iso_date(getattr(g, "start_date", None) or getattr(g, "start_time", None)),
            "home_team": getattr(g, "home_team", ""),
            "away_team": getattr(g, "away_team", ""),
            "home_points": hp,
            "away_points": ap,
        })
    finals_df = pd.DataFrame(finals)

    df = preds.merge(finals_df, on=["week","date","home_team","away_team"], how="left")

    # Grade value-side result  ✅
    res_side = df.apply(lambda r: bet_result_value_side(
        r.get("home_points"), r.get("away_points"),
        _safe_float(r.get("market_spread_book")), _safe_float(r.get("model_spread_book"))
    ), axis=1, result_type="expand")
    df["bet_result_value"] = res_side[0]
    df["bet_side_value"] = res_side[1]

    # Legacy (optional): home-only grading for audit
    def _ats_home(r):
        hp, ap, mkt = r.get("home_points"), r.get("away_points"), _safe_float(r.get("market_spread_book"))
        if hp is None or ap is None or not math.isfinite(mkt): return ""
        diff = float(hp) - float(ap)
        delta = diff + mkt
        if abs(delta) < 1e-9: return "PUSH"
        return "CORRECT" if delta > 0 else "INCORRECT"
    df["home_result_legacy"] = df.apply(_ats_home, axis=1)

    # Write predictions (two common names)
    p1 = os.path.join(out_dir, "upa_predictions_2024_backtest.csv")
    p2 = os.path.join(out_dir, "backtest_predictions_2024.csv")
    df.to_csv(p1, index=False)
    df.to_csv(p2, index=False)

    # Summary by week
    recs = []
    for wk, grp in df.groupby("week"):
        w = (grp["bet_result_value"] == "CORRECT").sum()
        l = (grp["bet_result_value"] == "INCORRECT").sum()
        p = (grp["bet_result_value"] == "PUSH").sum()
        tot = w + l
        hit = round((w / tot) * 100, 1) if tot else None
        recs.append({"week": wk, "wins": int(w), "losses": int(l), "pushes": int(p), "hit_pct": hit})
    pd.DataFrame(recs).sort_values("week").to_csv(os.path.join(out_dir, "backtest_summary_2024.csv"), index=False)

    print(f"[backtest {year_bt}] wrote: {p1}, {p2}, backtest_summary_2024.csv")

# -----------------------------
# Module: Outputs (live)
# -----------------------------
def write_team_inputs_csv(df_inputs: pd.DataFrame):
    out_csv = os.path.join(DATA_DIR, "upa_team_inputs_datadriven_v0.csv")
    df_inputs.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({df_inputs.shape[0]} rows)")

def write_schedule_csv(df_sched: pd.DataFrame):
    out_csv = os.path.join(DATA_DIR, "cfb_schedule.csv")
    df_sched.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({df_sched.shape[0]} rows)")

def write_predictions_and_edge(df_pred: pd.DataFrame):
    # Primary predictions
    out_pred = os.path.join(DATA_DIR, "upa_predictions.csv")
    df_pred.to_csv(out_pred, index=False)
    print(f"Wrote {out_pred} ({df_pred.shape[0]} rows)")

    # Live Edge (simple export of top edges/value)
    live_cols = [
        "week","date","away_team","home_team","neutral_site",
        "model_spread_book","market_spread_book","expected_market_spread_book",
        "edge_points_book","value_points_book","qualified_edge_flag"
    ]
    live = df_pred[live_cols].copy()
    out_edge = os.path.join(DATA_DIR, "live_edge_report.csv")
    live.to_csv(out_edge, index=False)
    print(f"Wrote {out_edge} ({live.shape[0]} rows)")

def write_status(year: int, df_inputs: pd.DataFrame, df_sched: pd.DataFrame, df_pred: pd.DataFrame):
    start = datetime.now(timezone.utc)
    status = {
        "generated_at_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": year,
        "teams": int(df_inputs.shape[0]),
        "games": int(df_sched.shape[0]),
        "predictions": int(df_pred.shape[0]),
        "fields_predictions": list(df_pred.columns),
        "next_run_eta_utc": (start + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(os.path.join(DATA_DIR, "status.json"), "w") as f:
        json.dump(status, f, indent=2)
    print("Wrote data/status.json")

# -----------------------------
# Main
# -----------------------------
def main():
    print(f"[live] building team inputs for {YEAR} ...")
    inputs_live = build_team_inputs(YEAR)
    write_team_inputs_csv(inputs_live)

    print(f"[live] building schedule + market for {YEAR} ...")
    sched_live = build_schedule_with_market(YEAR)
    write_schedule_csv(sched_live)

    print(f"[live] building predictions for {YEAR} ...")
    preds_live = build_predictions_for_year(YEAR, inputs_live)
    write_predictions_and_edge(preds_live)
    write_status(YEAR, inputs_live, sched_live, preds_live)

    if BACKTEST_YEAR:
        print(f"[backtest] running backtest for {BACKTEST_YEAR} ...")
        inputs_bt = build_team_inputs(BACKTEST_YEAR)
        run_backtest(BACKTEST_YEAR, inputs_bt)

if __name__ == "__main__":
    main()