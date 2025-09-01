
import os, json, sys
from datetime import datetime, timedelta, timezone
import pandas as pd

try:
    import cfbd
except Exception as e:
    print("ERROR: cfbd package not available. Ensure Actions installed 'cfbd'.", file=sys.stderr)
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

EXPECTED_RETURNING_COLS = [
    "team","conference",
    "wrps_offense_percent","wrps_defense_percent","wrps_overall_percent","wrps_percent_0_100"
]

def df_from_returning():
    print("Pulling returning production…", flush=True)

    fbs = teams_api.get_fbs_teams(year=YEAR)
    team_conf = {t.school: (t.conference or "FBS") for t in fbs}
    conferences = sorted({t.conference for t in fbs if t.conference})

    rows = []
    for conf in conferences:
        try:
            items = players_api.get_returning_production(year=YEAR, conference=conf)
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

    def normalize_percent(x):
        if pd.isna(x): return None
        try:
            x = float(x)
        except Exception:
            return None
        return x*100.0 if x <= 1.0 else x

    for src, out in [("_offense","wrps_offense_percent"),
                     ("_defense","wrps_defense_percent"),
                     ("_overall","wrps_overall_percent")]:
        if src in df.columns:
            df[out] = df[src].apply(normalize_percent)

    need_proxy = any((c not in df.columns) or df[c].isna().all()
                     for c in ["wrps_offense_percent","wrps_defense_percent","wrps_overall_percent"])

    if need_proxy:
        for col in ["_ppa_tot","_ppa_off","_ppa_def"]:
            if col not in df.columns:
                df[col] = None

        def scale(series):
            s = pd.to_numeric(series, errors="coerce")
            if s.notna().sum() == 0:
                return pd.Series([None]*len(s), index=s.index)
            mn, mx = float(s.min()), float(s.max())
            if mx == mn:
                return pd.Series([50.0]*len(s), index=s.index)
            return (s - mn) / (mx - mn) * 100.0

        if "wrps_overall_percent" not in df.columns or df["wrps_overall_percent"].isna().all():
            df["wrps_overall_percent"] = scale(df["_ppa_tot"])
        if "wrps_offense_percent" not in df.columns or df["wrps_offense_percent"].isna().all():
            df["wrps_offense_percent"] = scale(df["_ppa_off"])
        if "wrps_defense_percent" not in df.columns or df["wrps_defense_percent"].isna().all():
            df["wrps_defense_percent"] = scale(df["_ppa_def"])

    df["wrps_percent_0_100"] = pd.to_numeric(df.get("wrps_overall_percent"), errors="coerce").round(1)

    for col in EXPECTED_RETURNING_COLS:
        if col not in df.columns:
            df[col] = None

    if "conference" not in df.columns or df["conference"].isna().any():
        df["conference"] = df["team"].map(team_conf).fillna(df.get("conference")).fillna("FBS")

    return df[EXPECTED_RETURNING_COLS].copy()

def df_from_talent():
    print("Pulling Team Talent composite…", flush=True)
    try:
        items = teams_api.get_talent(year=YEAR)
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

def df_prev_season_sos_rank():
    print("Computing previous-season SOS rank via SRS…", flush=True)
    try:
        srs = ratings_api.get_srs(year=PRIOR)
    except Exception as e:
        print(f"[warn] srs fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})

    srs_map = {x.team: float(x.rating or 0) for x in srs}

    try:
        games = games_api.get_games(year=PRIOR, season_type="both")
    except Exception as e:
        print(f"[warn] games fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})

    from collections import defaultdict
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

def df_transfer_portal():
    print("Summarizing transfer portal (net)…", flush=True)
    try:
        portal = players_api.get_transfer_portal(year=YEAR)
    except Exception as e:
        print(f"[warn] portal fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})

    from collections import defaultdict
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

def main():
    start = datetime.now(timezone.utc)

    rp = df_from_returning()
    talent = df_from_talent()
    sos = df_prev_season_sos_rank()
    portal = df_transfer_portal()

    df = rp.merge(talent, on="team", how="left")            .merge(sos, on="team", how="left")            .merge(portal, on="team", how="left")

    if "conference" in df.columns:
        df.sort_values(["conference","team"], inplace=True, ignore_index=True)
    else:
        df.sort_values(["team"], inplace=True, ignore_index=True)

    out_csv = os.path.join(DATA_DIR, "upa_team_inputs_datadriven_v0.csv")
    df.to_csv(out_csv, index=False)

    status = {
        "generated_at_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": YEAR,
        "teams": int(df.shape[0]),
        "fields": list(df.columns),
        "next_run_eta_utc": (start + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(os.path.join(DATA_DIR, "status.json"), "w") as f:
        json.dump(status, f, indent=2)

    # Optional predictions generation if schedule exists at data/cfb_schedule.csv
    sched_path = os.path.join(DATA_DIR, "cfb_schedule.csv")
    preds_csv  = os.path.join(DATA_DIR, "upa_predictions.csv")
    edge_csv   = os.path.join(DATA_DIR, "live_edge_report.csv")
    try:
        if os.path.exists(sched_path):
            sc = pd.read_csv(sched_path)
            sc = sc.rename(columns={
                "home": "home_team", "away": "away_team",
                "homeTeam": "home_team", "awayTeam": "away_team",
                "neutral": "neutral_site", "is_neutral_site": "neutral_site"
            })
            base = df.copy()
            for need in ["wrps_percent_0_100","talent_score_0_100","portal_net_0_100"]:
                if need not in base.columns:
                    base[need] = 50.0
            if "prev_season_sos_rank_1_133" in base.columns and base["prev_season_sos_rank_1_133"].notna().any():
                sos = pd.to_numeric(base["prev_season_sos_rank_1_133"], errors="coerce")
                base["sos_0_100"] = (1 - (sos - 1) / (sos.max() - 1 if sos.max() > 1 else 1)) * 100.0
            else:
                base["sos_0_100"] = 50.0
            base["team_power_0_100"] = (
                0.35*pd.to_numeric(base["wrps_percent_0_100"], errors="coerce").fillna(50.0) +
                0.35*pd.to_numeric(base["talent_score_0_100"], errors="coerce").fillna(50.0) +
                0.15*pd.to_numeric(base["portal_net_0_100"], errors="coerce").fillna(50.0) +
                0.15*pd.to_numeric(base["sos_0_100"], errors="coerce").fillna(50.0)
            )
            base["team_rating"] = (base["team_power_0_100"] - 50.0) * 0.5
            rating = base.set_index("team")["team_rating"].to_dict()
            confmap = base.set_index("team")["conference"].to_dict()
            HFA = 2.0

            out_rows = []
            for _, r in sc.iterrows():
                ht, at = r.get("home_team"), r.get("away_team")
                if pd.isna(ht) or pd.isna(at): continue
                hr = float(rating.get(ht, 0.0)); ar = float(rating.get(at, 0.0))
                neutral = bool(int(r.get("neutral_site", 0))) if str(r.get("neutral_site","")).strip() != "" else False
                hfa = 0.0 if neutral else HFA
                model = (hr + hfa) - ar
                market = pd.to_numeric(r.get("market_spread", None), errors="coerce")
                edge = model - market if pd.notna(market) else None
                out_rows.append({
                    "week": r.get("week", None),
                    "date": r.get("date", None),
                    "home_team": ht,
                    "away_team": at,
                    "home_conf": confmap.get(ht, None),
                    "away_conf": confmap.get(at, None),
                    "neutral_site": int(neutral),
                    "model_spread_home": round(model, 2),
                    "market_spread_home": market if pd.notna(market) else None,
                    "edge_points": round(edge, 2) if edge is not None else None
                })
            pred_df = pd.DataFrame(out_rows)
            if not pred_df.empty:
                pred_df.to_csv(preds_csv, index=False)
                le = pred_df.dropna(subset=["edge_points"]).sort_values(by="edge_points", key=lambda s: s.abs(), ascending=False)
                le[["week","date","home_team","away_team","edge_points"]].head(200).to_csv(edge_csv, index=False)
    except Exception as e:
        print(f"[warn] predictions generation skipped/failed: {e}", file=sys.stderr)

    print(f"Wrote {out_csv} with {df.shape[0]} rows.", flush=True)

if __name__ == "__main__":
    main()
