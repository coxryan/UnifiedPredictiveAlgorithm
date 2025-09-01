import os
import sys
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import time

# -----------------------------
# CFBD client setup
# -----------------------------
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
    "team", "conference",
    "wrps_offense_percent", "wrps_defense_percent", "wrps_overall_percent",
    "wrps_percent_0_100"
]

# -----------------------------
# Returning Production
# -----------------------------
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
                "conference": getattr(it, "conference", None) or team_conf.get(getattr(it, "team", None), "FBS"),
                "_overall": getattr(it, "overall", None),
                "_offense": getattr(it, "offense", None),
                "_defense": getattr(it, "defense", None) or getattr(it, "defensive", None),
                "_ppa_tot": getattr(it, "total_ppa", None),
                "_ppa_off": getattr(it, "total_offense_ppa", None) or (
                    getattr(it, "total_passing_ppa", 0) + getattr(it, "total_rushing_ppa", 0)
                ),
                "_ppa_def": getattr(it, "total_defense_ppa", None) \
                            or getattr(it, "total_defensive_ppa", None) \
                            or getattr(it, "defense_ppa", None) \
                            or getattr(it, "defensive_ppa", None),
            }
            rows.append(rec)

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])

    if df.empty:
        print("[warn] returning production endpoint returned 0 records; using neutral 50s fallback.", flush=True)
        teams = sorted(team_conf.keys())
        df = pd.DataFrame({
            "team": teams,
            "conference": [team_conf.get(t, "FBS") for t in teams],
            "_overall": 50.0, "_offense": 50.0, "_defense": 50.0
        })

    def normalize_percent(x):
        if pd.isna(x): return None
        try:
            x = float(x)
        except Exception:
            return None
        return x * 100.0 if x <= 1.0 else x

    for src, out in [
        ("_offense", "wrps_offense_percent"),
        ("_defense", "wrps_defense_percent"),
        ("_overall", "wrps_overall_percent")
    ]:
        if src in df.columns:
            df[out] = df[src].apply(normalize_percent)

    def scale(series):
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() == 0:
            return pd.Series([None] * len(s), index=s.index)
        mn, mx = float(s.min()), float(s.max())
        if mx == mn:
            return pd.Series([50.0] * len(s), index=s.index)
        return (s - mn) / (mx - mn) * 100.0

    # Fill via PPA scaling where needed
    if "wrps_overall_percent" not in df.columns or df["wrps_overall_percent"].isna().all():
        df["wrps_overall_percent"] = scale(df["_ppa_tot"])
    if "wrps_offense_percent" not in df.columns or df["wrps_offense_percent"].isna().all():
        df["wrps_offense_percent"] = scale(df["_ppa_off"])
    if "wrps_defense_percent" not in df.columns or df["wrps_defense_percent"].isna().all():
        df["wrps_defense_percent"] = scale(df["_ppa_def"])

    df["wrps_percent_0_100"] = pd.to_numeric(df.get("wrps_overall_percent"), errors="coerce").round(1)

    # Robust DEF fallback if still missing
    if df["wrps_defense_percent"].isna().all():
        ovr = pd.to_numeric(df.get("wrps_overall_percent"), errors="coerce")
        off = pd.to_numeric(df.get("wrps_offense_percent"), errors="coerce")
        if ovr.notna().any() and off.notna().any():
            df["wrps_defense_percent"] = (2 * ovr - off).clip(0, 100)

    for col in EXPECTED_RETURNING_COLS:
        if col not in df.columns:
            df[col] = None

    if "conference" not in df.columns or df["conference"].isna().any():
        df["conference"] = df["team"].map(team_conf).fillna(df.get("conference")).fillna("FBS")

    return df[EXPECTED_RETURNING_COLS].copy()

# -----------------------------
# Team Talent
# -----------------------------
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
    df["talent_score_0_100"] = 50.0 if mx == mn else ((df["talent"] - mn) / (mx - mn) * 100.0).round(1)
    return df[["team", "talent_score_0_100"]]

# -----------------------------
# Previous Season SOS (SRS + opp avg)
# -----------------------------
def df_prev_season_sos_rank():
    print("Computing previous-season SOS rank via SRS…", flush=True)
    try:
        srs = ratings_api.get_srs(year=PRIOR)
        games = games_api.get_games(year=PRIOR, season_type="both")
    except Exception as e:
        print(f"[warn] srs or games fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})

    srs_map = {x.team: float(x.rating or 0) for x in srs}

    from collections import defaultdict
    opps = defaultdict(list)
    for g in games:
        ht, at = getattr(g, "home_team", None), getattr(g, "away_team", None)
        if ht in srs_map and at in srs_map:
            opps[ht].append(srs_map[at])
            opps[at].append(srs_map[ht])

    rows = [{"team": t, "sos_value": sum(v) / len(v)} for t, v in opps.items() if v]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})

    df["prev_season_sos_rank_1_133"] = df["sos_value"].rank(ascending=False, method="min").astype(int)
    return df[["team", "prev_season_sos_rank_1_133"]]

# -----------------------------
# Transfer Portal (net)
# -----------------------------
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
        val = getattr(p, "rating", None) or getattr(p, "stars", None) or 1.0
        try:
            val = float(val)
        except Exception:
            val = 1.0
        if to_team:
            incoming[to_team] += 1; rating_in[to_team] += val
        if from_team:
            outgoing[from_team] += 1; rating_out[from_team] += val

    teams = set(list(incoming.keys()) + list(outgoing.keys()))
    rows = [{
        "team": t,
        "portal_net_count": incoming[t] - outgoing[t],
        "portal_net_value": rating_in[t] - rating_out[t]
    } for t in teams]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})

    def scale(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() == 0:
            return pd.Series([50.0] * len(s), index=s.index)
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series([50.0] * len(s), index=s.index)
        return (s - mn) / (mx - mn) * 100.0

    df["portal_net_0_100"] = (0.5 * scale(df["portal_net_count"]) + 0.5 * scale(df["portal_net_value"])).round(1)
    return df[["team", "portal_net_0_100", "portal_net_count", "portal_net_value"]]

# -----------------------------
# Schedule builder (SDK week-by-week, no division arg)
# -----------------------------
def build_schedule_df() -> pd.DataFrame:
    """
    Build the schedule week-by-week using the CFBD SDK GamesApi.get_games.
    We try season_type='regular' first, then 'both'.
    Logs loudly so we can see exactly what's returned per week.
    Returns columns: week,date,home_team,away_team,neutral_site,market_spread
    """
    print("BEGIN schedule: SDK week-by-week get_games()", flush=True)

    def safe_date(g) -> str:
        # g.start_date may be a datetime OR a string, depending on SDK version
        sd = getattr(g, "start_date", None) or getattr(g, "start_time", None)
        if isinstance(sd, datetime):
            return sd.date().isoformat()
        try:
            s = str(sd)
            return s[:10] if s else ""
        except Exception:
            return ""

    def rows_from_games(glist):
        rows = []
        for g in glist or []:
            ht = getattr(g, "home_team", None)
            at = getattr(g, "away_team", None)
            if not ht or not at:
                continue
            rows.append({
                "week": getattr(g, "week", None),
                "date": safe_date(g),
                "home_team": ht,
                "away_team": at,
                "neutral_site": 1 if getattr(g, "neutral_site", False) else 0,
                "market_spread": None,
            })
        return rows

    all_rows = []
    try:
        for wk in range(0, 22):  # Week 0 .. 21 buffer
            last_err = None
            for attempt in range(4):
                try:
                    games = games_api.get_games(year=YEAR, week=wk, season_type="regular") or []
                    if not games:
                        games = games_api.get_games(year=YEAR, week=wk, season_type="both") or []
                    print(f"DEBUG SDK games week {wk}: {len(games)}", flush=True)
                    all_rows += rows_from_games(games)
                    break
                except Exception as e:
                    last_err = e
                    backoff = 0.6 * (attempt + 1)
                    print(f"[warn] SDK get_games week {wk} attempt {attempt+1}/4 failed: {e}; retrying in {backoff:.1f}s", file=sys.stderr)
                    time.sleep(backoff)
            else:
                print(f"[warn] SDK get_games week {wk} failed after retries: {last_err}", file=sys.stderr)
    except Exception as e:
        print(f"[error] schedule loop crashed: {e}", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    total = len(df)
    print(f"END schedule: SDK built total rows = {total}", flush=True)

    cols = ["week","date","home_team","away_team","neutral_site","market_spread"]

    # Optional fallback to HTTP if SDK returns 0 — only if you explicitly allow it.
    allow_http = os.environ.get("UPA_ALLOW_HTTP_FALLBACK", "0").strip() in ("1","true","yes")
    if total == 0 and allow_http:
        print("[info] SDK returned 0 rows; attempting HTTP fallback (UPA_ALLOW_HTTP_FALLBACK=1).", flush=True)
        try:
            import requests
            base_url = "https://api.collegefootballdata.com/games"
            headers = {"Authorization": f"Bearer {BEARER}", "Accept": "application/json"}
            http_rows = []
            for wk in range(0, 22):
                r = requests.get(base_url, headers=headers, params={
                    "year": str(YEAR),
                    "week": str(wk),
                    "seasonType": "regular"
                }, timeout=30)
                if r.status_code == 200:
                    for g in r.json() or []:
                        ht, at = g.get("home_team"), g.get("away_team")
                        if not ht or not at: continue
                        http_rows.append({
                            "week": g.get("week"),
                            "date": (g.get("start_date") or "")[:10],
                            "home_team": ht,
                            "away_team": at,
                            "neutral_site": 1 if g.get("neutral_site") else 0,
                            "market_spread": None,
                        })
                else:
                    print(f"[warn] HTTP fallback week {wk} status {r.status_code}", file=sys.stderr)
            df = pd.DataFrame(http_rows)
            total = len(df)
            print(f"HTTP fallback built rows = {total}", flush=True)
        except Exception as e:
            print(f"[warn] HTTP fallback failed: {e}", file=sys.stderr)

    if total == 0:
        # Sanity check: can token pull prior season?
        try:
            prior = games_api.get_games(year=YEAR-1, week=1, season_type="regular") or []
            print(f"DEBUG token sanity: prior season week 1 via SDK returned {len(prior)} games", flush=True)
        except Exception as e:
            print(f"[warn] prior-season sanity check failed: {e}", file=sys.stderr)
        return pd.DataFrame(columns=cols)

    df = df[cols]
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["neutral_site"] = pd.to_numeric(df["neutral_site"], errors="coerce").fillna(0).astype(int)
    return df
    """
    Build schedule using GamesApi.get_games week-by-week with retries.
    season_type='regular', fallback 'both'. No division arg.
    Returns: columns week,date,home_team,away_team,neutral_site,market_spread
    """
def rows_from_games(glist):
    rows = []
    for g in glist or []:
        ht = getattr(g, "home_team", None)
        at = getattr(g, "away_team", None)
        if not ht or not at:
            continue

        # --- robust date extraction ---
        sd = getattr(g, "start_date", None) or getattr(g, "start_time", None)
        if isinstance(sd, datetime):
            date_str = sd.date().isoformat()
        else:
            # cfbd may return ISO string; fall back safely
            try:
                date_str = str(sd)[:10] if sd else ""
            except Exception:
                date_str = ""

        rows.append({
            "week": getattr(g, "week", None),
            "date": date_str,
            "home_team": ht,
            "away_team": at,
            "neutral_site": 1 if getattr(g, "neutral_site", False) else 0,
            "market_spread": None,
        })
    return rows

    all_rows = []
    for wk in range(0, 22):  # Week 0 .. Week 21 buffer
        last_err = None
        for attempt in range(4):
            try:
                games = games_api.get_games(year=YEAR, week=wk, season_type="regular") or []
                if not games:
                    games = games_api.get_games(year=YEAR, week=wk, season_type="both") or []
                print(f"DEBUG SDK games week {wk}: {len(games)}", flush=True)
                all_rows += rows_from_games(games)
                break
            except Exception as e:
                last_err = e
                backoff = 0.6 * (attempt + 1)
                print(f"[warn] SDK get_games week {wk} attempt {attempt+1}/4 failed: {e}; retrying in {backoff:.1f}s", file=sys.stderr)
                time.sleep(backoff)
        else:
            print(f"[warn] SDK get_games week {wk} failed after retries: {last_err}", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    print(f"DEBUG SDK built schedule total rows: {len(df)}", flush=True)

    cols = ["week", "date", "home_team", "away_team", "neutral_site", "market_spread"]
    if df.empty:
        # Sanity check: can token pull prior season?
        try:
            prior = games_api.get_games(year=YEAR - 1, week=1, season_type="regular") or []
            print(f"DEBUG token sanity: prior season week 1 returned {len(prior)} games", flush=True)
        except Exception as e:
            print(f"[warn] prior-season sanity check failed: {e}", file=sys.stderr)
        return pd.DataFrame(columns=cols)

    df = df[cols]
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["neutral_site"] = pd.to_numeric(df["neutral_site"], errors="coerce").fillna(0).astype(int)
    return df

# -----------------------------
# Predictions / Live Edge
# -----------------------------
def build_predictions(team_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    base = team_df.copy()

    for need in ["wrps_percent_0_100", "talent_score_0_100", "portal_net_0_100"]:
        if need not in base.columns:
            base[need] = 50.0

    if "prev_season_sos_rank_1_133" in base.columns and \
       pd.to_numeric(base["prev_season_sos_rank_1_133"], errors="coerce").notna().any():
        sosr = pd.to_numeric(base["prev_season_sos_rank_1_133"], errors="coerce")
        base["sos_0_100"] = (1 - (sosr - 1) / (sosr.max() - 1 if sosr.max() > 1 else 1)) * 100.0
    else:
        base["sos_0_100"] = 50.0

    base["team_power_0_100"] = (
        0.35 * pd.to_numeric(base["wrps_percent_0_100"], errors="coerce").fillna(50.0) +
        0.35 * pd.to_numeric(base["talent_score_0_100"], errors="coerce").fillna(50.0) +
        0.15 * pd.to_numeric(base["portal_net_0_100"], errors="coerce").fillna(50.0) +
        0.15 * pd.to_numeric(base["sos_0_100"], errors="coerce").fillna(50.0)
    )
    base["team_rating"] = (base["team_power_0_100"] - 50.0) * 0.5

    rating = base.set_index("team")["team_rating"].to_dict()
    confmap = base.set_index("team")["conference"].to_dict()

    HFA = 2.0
    out_rows = []
    for _, r in schedule_df.iterrows():
        ht, at = r.get("home_team"), r.get("away_team")
        if pd.isna(ht) or pd.isna(at):
            continue
        hr = float(rating.get(ht, 0.0))
        ar = float(rating.get(at, 0.0))
        neutral = bool(int(r.get("neutral_site", 0))) if str(r.get("neutral_site", "")).strip() != "" else False
        hfa = 0.0 if neutral else HFA
        model_spread = (hr + hfa) - ar
        market = pd.to_numeric(r.get("market_spread", None), errors="coerce")
        edge = model_spread - market if pd.notna(market) else None

        out_rows.append({
            "week": r.get("week"),
            "date": r.get("date"),
            "home_team": ht,
            "away_team": at,
            "home_conf": confmap.get(ht),
            "away_conf": confmap.get(at),
            "neutral_site": int(neutral),
            "model_spread_home": round(model_spread, 2),
            "market_spread_home": market if pd.notna(market) else None,
            "edge_points": round(edge, 2) if edge is not None else None
        })

    return pd.DataFrame(out_rows)

# -----------------------------
# Main
# -----------------------------
def main():
    start = datetime.now(timezone.utc)

    # Collect modules
    rp = df_from_returning()
    talent = df_from_talent()
    sos = df_prev_season_sos_rank()
    portal = df_transfer_portal()

    # Merge on team
    team_df = rp.merge(talent, on="team", how="left") \
                .merge(sos, on="team", how="left") \
                .merge(portal, on="team", how="left")

    if "conference" in team_df.columns:
        team_df.sort_values(["conference", "team"], inplace=True, ignore_index=True)
    else:
        team_df.sort_values(["team"], inplace=True, ignore_index=True)

    # Write team inputs
    inputs_csv = os.path.join(DATA_DIR, "upa_team_inputs_datadriven_v0.csv")
    team_df.to_csv(inputs_csv, index=False)
    print(f"DEBUG wrote team inputs CSV to {inputs_csv} with {team_df.shape[0]} rows", flush=True)

    # Status file for dashboard
    status = {
        "generated_at_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": YEAR,
        "teams": int(team_df.shape[0]),
        "fields": list(team_df.columns),
        "next_run_eta_utc": (start + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(DATA_DIR, "status.json"), "w") as f:
        json.dump(status, f, indent=2)

    # Build schedule and persist
    sched_path = os.path.join(DATA_DIR, "cfb_schedule.csv")
    schedule_df = build_schedule_df()

    cols_sched = ["week", "date", "home_team", "away_team", "neutral_site", "market_spread"]
    if schedule_df is None or schedule_df.empty:
        pd.DataFrame(columns=cols_sched).to_csv(sched_path, index=False)
        print(f"[warn] wrote header-only schedule CSV to {sched_path} (0 rows)", file=sys.stderr)
        print("[warn] no schedule available; skipping predictions/edge.", file=sys.stderr)
        # Still exit successfully so Pages deploys status + inputs
        print(f"Wrote {inputs_csv} with {team_df.shape[0]} rows.", flush=True)
        return
    else:
        schedule_df.to_csv(sched_path, index=False)
        print(f"DEBUG wrote schedule CSV to {sched_path} with {len(schedule_df)} rows", flush=True)

    # Predictions + live edge
    preds_df = build_predictions(team_df, schedule_df)
    preds_csv = os.path.join(DATA_DIR, "upa_predictions.csv")
    preds_df.to_csv(preds_csv, index=False)
    print(f"DEBUG wrote predictions CSV to {preds_csv} with {preds_df.shape[0]} rows", flush=True)

    edge_csv = os.path.join(DATA_DIR, "live_edge_report.csv")
    if "edge_points" in preds_df.columns:
        edge = preds_df.dropna(subset=["edge_points"]).sort_values(
            by="edge_points", key=lambda s: s.abs(), ascending=False
        )
        edge[["week", "date", "home_team", "away_team", "edge_points"]].head(200).to_csv(edge_csv, index=False)
        print(f"DEBUG wrote live edge CSV to {edge_csv} with {min(200, edge.shape[0])} rows", flush=True)
    else:
        # Shouldn't happen, but keep the portal consistent
        pd.DataFrame(columns=["week", "date", "home_team", "away_team", "edge_points"]).to_csv(edge_csv, index=False)
        print(f"[warn] preds_df has no edge_points; wrote header-only {edge_csv}", file=sys.stderr)

    # Optional: Google Sheets upsert
    SHEET_ID = os.environ.get("SHEET_ID", "").strip()
    SA = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if SHEET_ID and SA and os.path.exists(SA):
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_file(SA, scopes=scopes)
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(SHEET_ID)
            try:
                ws = sh.worksheet("Team Inputs")
            except Exception:
                ws = sh.add_worksheet(title="Team Inputs", rows=str(len(team_df) + 10), cols=str(len(team_df.columns) + 5))
            ws.clear()
            ws.update([team_df.columns.tolist()] + team_df.fillna("").values.tolist())
            print("Sheets updated: Team Inputs", flush=True)
        except Exception as e:
            print(f"[warn] Sheets update skipped/failed: {e}", file=sys.stderr)

    print(f"Wrote {inputs_csv} with {team_df.shape[0]} rows.", flush=True)

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    main()