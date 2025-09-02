import os
import sys
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import pandas as pd
import numpy as np

# ----------------------- CFBD SDK setup -----------------------
try:
    import cfbd
except Exception as e:
    print("ERROR: cfbd package not available. Ensure Actions installs 'cfbd'.", file=sys.stderr)
    raise

BEARER = os.environ.get("BEARER_TOKEN", "").strip()
if not BEARER:
    print("ERROR: Missing BEARER_TOKEN secret for CollegeFootballData API.", file=sys.stderr)
    sys.exit(1)

YEAR = int(os.environ.get("UPA_YEAR", "2025"))
PRIOR = YEAR - 1
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

configuration = cfbd.Configuration(access_token=BEARER)
api_client = cfbd.ApiClient(configuration)
teams_api = cfbd.TeamsApi(api_client)
players_api = cfbd.PlayersApi(api_client)
ratings_api = cfbd.RatingsApi(api_client)
games_api = cfbd.GamesApi(api_client)
lines_api = cfbd.BettingApi(api_client)  # for market spreads

EXPECTED_RETURNING_COLS = [
    "team","conference",
    "wrps_offense_percent","wrps_defense_percent","wrps_overall_percent","wrps_percent_0_100"
]

# ----------------------- helpers -----------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def _to_float(x):
    try:
        if x is None or x == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _normalize_pct(x):
    if pd.isna(x): return np.nan
    try:
        v = float(x)
    except Exception:
        return np.nan
    return v*100.0 if v <= 1.0 else v

# ----------------------- data sources -----------------------
def df_from_returning(year: int):
    print("Pulling returning production…", flush=True)

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
            rec = {
                "team": getattr(it, "team", None),
                "conference": getattr(it, "conference", None) or team_conf.get(getattr(it,"team",None), ""),
                "_overall": getattr(it, "overall", None),
                "_offense": getattr(it, "offense", None),
                "_defense": getattr(it, "defense", None),
                "_ppa_tot": getattr(it, "total_ppa", None),
                "_ppa_off": getattr(it, "total_offense_ppa", None) or (getattr(it, "total_passing_ppa", 0) + getattr(it, "total_rushing_ppa", 0)),
                "_ppa_def": getattr(it, "total_defense_ppa", None) or getattr(it, "total_defensive_ppa", None),
            }
            rows.append(rec)

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])
    for src, out in [("_offense","wrps_offense_percent"),
                     ("_defense","wrps_defense_percent"),
                     ("_overall","wrps_overall_percent")]:
        df[out] = df[src].apply(_normalize_pct)

    def scale(series):
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() == 0:
            return pd.Series([np.nan]*len(s), index=s.index)
        mn, mx = float(s.min()), float(s.max())
        if mx == mn:
            return pd.Series([50.0]*len(s), index=s.index)
        return (s - mn) / (mx - mn) * 100.0

    # fallbacks when percent columns are missing/empty
    if df["wrps_overall_percent"].isna().all():
        df["wrps_overall_percent"] = scale(df["_ppa_tot"])
    if df["wrps_offense_percent"].isna().all():
        df["wrps_offense_percent"] = scale(df["_ppa_off"])
    if df["wrps_defense_percent"].isna().all():
        df["wrps_defense_percent"] = scale(df["_ppa_def"])

    df["wrps_percent_0_100"] = pd.to_numeric(df.get("wrps_overall_percent"), errors="coerce").round(1)
    for col in EXPECTED_RETURNING_COLS:
        if col not in df.columns:
            df[col] = np.nan
    if "conference" not in df.columns or df["conference"].isna().any():
        df["conference"] = df["team"].map(team_conf).fillna(df.get("conference")).fillna("FBS")

    return df[EXPECTED_RETURNING_COLS].copy()

def df_from_talent(year: int):
    print("Pulling Team Talent composite…", flush=True)
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
    if mx == mn:
        df["talent_score_0_100"] = 50.0
    else:
        df["talent_score_0_100"] = ((df["talent"] - mn) / (mx - mn) * 100.0).round(1)
    return df[["team", "talent_score_0_100"]]

def df_prev_season_sos_rank(prior_year: int):
    print("Computing previous-season SOS rank via SRS…", flush=True)
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
        if not ht or not at:
            continue
        if ht in srs_map and at in srs_map:
            opps[ht].append(srs_map[at])
            opps[at].append(srs_map[ht])

    rows = [{"team": t, "sos_value": sum(v)/len(v)} for t, v in opps.items() if v]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})

    df["prev_season_sos_rank_1_133"] = df["sos_value"].rank(ascending=False, method="min").astype(int)
    return df[["team","prev_season_sos_rank_1_133"]]

def df_transfer_portal(year: int):
    print("Summarizing transfer portal (net)…", flush=True)
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

    def scale(series):
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() == 0:
            return pd.Series([50.0]*len(s), index=s.index)
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series([50.0]*len(s), index=s.index)
        return (s - mn) / (mx - mn) * 100.0

    df["portal_net_0_100"] = (0.5*scale(df["portal_net_count"]) + 0.5*scale(df["portal_net_value"])).round(1)
    return df[["team","portal_net_0_100","portal_net_count","portal_net_value"]]

# ----------------------- model-building blocks -----------------------
def build_team_inputs(year: int):
    rp = df_from_returning(year)
    talent = df_from_talent(year)
    sos_prev = df_prev_season_sos_rank(year-1)
    portal = df_transfer_portal(year)
    df = rp.merge(talent, on="team", how="left") \
           .merge(sos_prev, on="team", how="left") \
           .merge(portal, on="team", how="left")

    if "conference" in df.columns:
        df.sort_values(["conference","team"], inplace=True, ignore_index=True)
    else:
        df.sort_values(["team"], inplace=True, ignore_index=True)

    # simple composite power (placeholder — your repo may add more advanced metrics)
    comps = []
    comps.append(pd.to_numeric(df.get("wrps_percent_0_100"), errors="coerce").fillna(50))
    comps.append(pd.to_numeric(df.get("talent_score_0_100"), errors="coerce").fillna(50))
    comps.append(100 - pd.to_numeric(df.get("prev_season_sos_rank_1_133"), errors="coerce").fillna(100) / 133.0 * 100.0)
    comps.append(pd.to_numeric(df.get("portal_net_0_100"), errors="coerce").fillna(50))
    df["team_power_0_100"] = np.vstack(comps).mean(axis=0).round(1)

    return df

def build_schedule(year: int, fbs_teams: set[str]):
    rows = []
    # pull regular + postseason for completeness
    for season_type in ["regular", "postseason"]:
        try:
            games = games_api.get_games(year=year, season_type=season_type)
        except Exception as e:
            print(f"[warn] games fetch failed ({season_type}): {e}", file=sys.stderr)
            continue
        for g in games or []:
            ht, at = getattr(g,"home_team",None), getattr(g,"away_team",None)
            if not ht or not at: 
                continue
            if ht not in fbs_teams or at not in fbs_teams:
                continue  # drop FCS or non-FBS
            gid = getattr(g, "id", None)
            rows.append({
                "game_id": int(gid) if gid is not None else None,
                "week": int(getattr(g,"week",0) or 0),
                "date": str(getattr(g,"start_date",""))[:10],
                "home_team": ht, "away_team": at,
                "neutral_site": 1 if getattr(g,"neutral_site",False) else 0,
                "home_points": _to_float(getattr(g,"home_points",None)),
                "away_points": _to_float(getattr(g,"away_points",None)),
                "played": bool(getattr(g,"completed", False)) or (_to_float(getattr(g,"home_points",None))==_to_float(getattr(g,"home_points",None)) and _to_float(getattr(g,"away_points",None))==_to_float(getattr(g,"away_points",None))),
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["game_id"])
    return df

def fetch_market_lines(year: int, game_ids: list[int] | None = None):
    """
    Pull consensus (median) closing spread for each game_id from CFBD betting API.
    Book-style convention: negative = home favored.
    """
    try:
        lines = lines_api.get_lines(year=year)
    except Exception as e:
        print(f"[warn] betting lines fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"game_id": [], "market_spread_book": []})

    by_game = defaultdict(list)
    for ln in lines or []:
        gid = getattr(ln, "id", None)
        if gid is None:
            continue
        for market in getattr(ln, "lines", []) or []:
            spread = getattr(market, "spread", None)
            home_away = getattr(market, "formattedSpread", None)  # not always present
            if spread is None:
                continue
            # CFBD spreads are "home spread" when available; we treat negative = home favored
            by_game[int(gid)].append(float(spread))

    rows = []
    for gid, arr in by_game.items():
        if not arr:
            continue
        med = float(np.median(arr))
        rows.append({"game_id": gid, "market_spread_book": med})
    df = pd.DataFrame(rows)
    if game_ids is not None:
        df = df[df["game_id"].isin(game_ids)]
    return df

def build_predictions(year: int, teams_df: pd.DataFrame, sched_df: pd.DataFrame):
    # map conferences
    conf_map = {r.team: r.conference for r in teams_df[["team","conference"]].itertuples(index=False)}
    sched_df = sched_df.copy()
    sched_df["home_conf"] = sched_df["home_team"].map(conf_map)
    sched_df["away_conf"] = sched_df["away_team"].map(conf_map)

    # basic power to score (book-style spread from home perspective: negative = home stronger)
    power = teams_df.set_index("team")["team_power_0_100"].to_dict()
    hfa_by_conf = defaultdict(lambda: 2.0)  # simple default HFA by conference
    sched_df["h_power"] = sched_df["home_team"].map(power).fillna(50)
    sched_df["a_power"] = sched_df["away_team"].map(power).fillna(50)
    sched_df["hfa_points"] = sched_df.apply(
        lambda r: 0.0 if int(r.get("neutral_site",0) or 0)==1 else hfa_by_conf[str(r.get("home_conf",""))],
        axis=1
    )
    # model spread (home): negative = home favored
    sched_df["model_spread_book"] = -1.0 * (sched_df["h_power"] - sched_df["a_power"]) / 5.0 - sched_df["hfa_points"]

    # market spread
    mkt = fetch_market_lines(year, list(sched_df["game_id"].dropna().astype(int)))
    sched_df = sched_df.merge(mkt, on="game_id", how="left")

    # expected market via simple isotonic-ish clamp (here linear as placeholder)
    # You can replace with your calibration fit saved from training.
    diff = (sched_df["model_spread_book"] - sched_df["market_spread_book"]).abs()
    # edge/value (book-style)
    sched_df["edge_points_book"] = sched_df["model_spread_book"] - sched_df["market_spread_book"]
    sched_df["expected_market_spread_book"] = sched_df["model_spread_book"] - np.clip(sched_df["edge_points_book"], -3.0, 3.0)
    sched_df["value_points_book"] = sched_df["market_spread_book"] - sched_df["expected_market_spread_book"]

    # qualification
    EDGE_MIN = float(os.environ.get("EDGE_MIN", "2.0"))
    VALUE_MIN = float(os.environ.get("VALUE_MIN", "1.0"))
    side_agree = np.sign(sched_df["model_spread_book"]) == np.sign(sched_df["expected_market_spread_book"])
    sched_df["qualified_edge_flag"] = ((side_agree) & (diff >= EDGE_MIN) & (sched_df["value_points_book"].abs() >= VALUE_MIN)).astype(int)

    # results vs actual
    def _winner(row):
        if not bool(row.get("played", False)):
            return ""
        ap, hp = _to_float(row.get("away_points")), _to_float(row.get("home_points"))
        if not np.isfinite(ap) or not np.isfinite(hp):
            return ""
        if hp > ap: return "HOME"
        if ap > hp: return "AWAY"
        return "PUSH"

    def _model_pick(row):
        s = _to_float(row.get("model_spread_book"))
        if not np.isfinite(s): return ""
        return "HOME" if s < 0 else "AWAY" if s > 0 else "PUSH"

    sched_df["actual_winner"] = sched_df.apply(_winner, axis=1)
    sched_df["model_pick"] = sched_df.apply(_model_pick, axis=1)
    def _graded(row):
        if row["actual_winner"] == "": return ""
        if row["model_pick"] == "": return ""
        if row["actual_winner"] == "PUSH": return "PUSH"
        return "CORRECT" if row["actual_winner"] == row["model_pick"] else "INCORRECT"
    sched_df["model_result"] = sched_df.apply(_graded, axis=1)

    return sched_df

def write_outputs(year: int, teams_df: pd.DataFrame, sched_df: pd.DataFrame, preds_df: pd.DataFrame):
    # team inputs
    teams_path = os.path.join(DATA_DIR, "upa_team_inputs_datadriven_v0.csv")
    teams_df.to_csv(teams_path, index=False)

    # schedule CSV
    sched_path = os.path.join(DATA_DIR, "cfb_schedule.csv")
    cols = ["game_id","week","date","away_team","home_team","neutral_site","away_points","home_points","played"]
    (sched_df[cols].sort_values(["week","date","home_team","away_team"]).reset_index(drop=True)
     ).to_csv(sched_path, index=False)

    # predictions
    preds_cols = [
        "game_id","week","date",
        "away_team","home_team","neutral_site",
        "away_points","home_points","played",
        "market_spread_book","model_spread_book","expected_market_spread_book",
        "edge_points_book","value_points_book",
        "qualified_edge_flag","model_result"
    ]
    preds_path = os.path.join(DATA_DIR, "upa_predictions.csv")
    (preds_df[preds_cols].sort_values(["week","date","home_team","away_team"]).reset_index(drop=True)
     ).to_csv(preds_path, index=False)

    # live edge (subset for UI)
    live_cols = ["week","date","away_team","home_team","model_spread_book","market_spread_book","edge_points_book","qualified_edge_flag"]
    live_path = os.path.join(DATA_DIR, "live_edge_report.csv")
    preds_df[live_cols].to_csv(live_path, index=False)

    # status
    status = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": year,
        "teams": int(teams_df.shape[0]),
        "games": int(sched_df.shape[0]),
        "pred_rows": int(preds_df.shape[0]),
        "next_run_eta_utc": (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(os.path.join(DATA_DIR, "status.json"), "w") as f:
        json.dump(status, f, indent=2)

# ----------------------- backtest -----------------------
def run_backtest(year: int, weeks: list[int]) -> dict:
    print(f"[backtest] {year} weeks {weeks}")
    teams_df = build_team_inputs(year)
    fbs_set = set(teams_df["team"])
    sched_df = build_schedule(year, fbs_set)
    sched_df = sched_df[sched_df["week"].astype(int).isin(weeks)]
    preds_df = build_predictions(year, teams_df, sched_df)

    # weekly hit summary
    rows = []
    for wk in sorted(preds_df["week"].astype(int).unique()):
        sub = preds_df[(preds_df["week"].astype(int)==wk) & (preds_df["played"]==True)]
        hit = (sub["model_result"]=="CORRECT").mean() if not sub.empty else np.nan
        rows.append({"week": int(wk), "rows": int(sub.shape[0]), "hit_rate": round(float(hit),3) if pd.notna(hit) else None})
    overall = (preds_df[preds_df["played"]==True]["model_result"]=="CORRECT").mean()
    rows.append({"week":"ALL","rows": int((preds_df["played"]==True).sum()), "hit_rate": round(float(overall),3)})

    out_dir = _ensure_dir(os.path.join(DATA_DIR, str(year)))
    preds_path = os.path.join(out_dir, f"upa_predictions_{year}_backtest.csv")
    summary_path = os.path.join(out_dir, f"backtest_summary_{year}.csv")
    preds_df.to_csv(preds_path, index=False)
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(f"[backtest done] year={year} overall_hit={overall:.3f}")
    return {"preds": preds_path, "summary": summary_path}

# ----------------------- entrypoint -----------------------
def main():
    # Always generate the 2024 backtest (weeks 1–15) into data/2024/*
    try:
        run_backtest(2024, list(range(1, 16)))
    except Exception as e:
        print(f"[warn] backtest 2024 failed: {e}", file=sys.stderr)

    # Live (2025) collection
    teams_df = build_team_inputs(YEAR)
    fbs_set = set(teams_df["team"])
    sched_df = build_schedule(YEAR, fbs_set)
    preds_df = build_predictions(YEAR, teams_df, sched_df)
    write_outputs(YEAR, teams_df, sched_df, preds_df)
    print("Collector completed.", flush=True)

if __name__ == "__main__":
    main()