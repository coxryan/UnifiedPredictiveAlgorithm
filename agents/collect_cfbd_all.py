# agents/collect_cfbd_all.py
import os
import sys
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np

try:
    import cfbd
except Exception as e:
    print("ERROR: cfbd package not available. Ensure Actions installed 'cfbd'.", file=sys.stderr)
    raise

# ---------------- Config ----------------
BEARER = os.environ.get("BEARER_TOKEN", "").strip()
if not BEARER:
    print("ERROR: Missing BEARER_TOKEN secret for CollegeFootballData API.", file=sys.stderr)
    sys.exit(1)

YEAR = int(os.environ.get("UPA_YEAR", "2025"))
PRIOR = YEAR - 1
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HFA_POINTS = 2.0             # home field advantage used in both pipelines
ALPHA_CLAMP = (0.8, 2.0)     # clamp for handicap calibration alpha
ALPHA2_CLAMP = (0.5, 3.0)    # clamp for advantage→market regression alpha

# ---------------- CFBD clients ----------------
configuration = cfbd.Configuration(access_token=BEARER)
api_client = cfbd.ApiClient(configuration)
teams_api = cfbd.TeamsApi(api_client)
players_api = cfbd.PlayersApi(api_client)
ratings_api = cfbd.RatingsApi(api_client)
games_api = cfbd.GamesApi(api_client)
bet_api = cfbd.BettingApi(api_client)

# ---------------- Helpers ----------------
def _normalize_percent(x):
    if pd.isna(x): return None
    try:
        x = float(x)
    except Exception:
        return None
    return x * 100.0 if x <= 1.0 else x

def _scale_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([50.0] * len(s), index=s.index)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100.0

def _zscore_clip(arr: np.ndarray, clip=2.5) -> np.ndarray:
    m = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        z = np.zeros_like(arr)
    else:
        z = (arr - m) / sd
    return np.clip(z, -clip, clip)

def _lin_reg(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return alpha (slope), beta (intercept) using OLS on finite pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return (1.0, 0.0)
    a, b = np.polyfit(x[mask], y[mask], 1)
    return float(a), float(b)

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# ---------------- Data pulls ----------------
def df_fbs_teams(year: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    fbs = teams_api.get_fbs_teams(year=year)
    rows = [{"team": t.school, "conference": getattr(t, "conference", None) or "FBS"} for t in fbs]
    df = pd.DataFrame(rows).drop_duplicates("team")
    conf_map = {r["team"]: r["conference"] for _, r in df.iterrows()}
    return df, conf_map

def df_returning(year: int, confs: List[str], team_conf: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for conf in confs:
        try:
            items = players_api.get_returning_production(year=year, conference=conf)
        except Exception as e:
            print(f"[warn] returning production fetch failed for {conf}: {e}", file=sys.stderr)
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
    # normalize percent fields when present
    for src, out in [("_offense","wrps_offense_percent"),
                     ("_defense","wrps_defense_percent"),
                     ("_overall","wrps_overall_percent")]:
        if src in df.columns:
            df[out] = df[src].apply(_normalize_percent)

    need_proxy = any((c not in df.columns) or df[c].isna().all()
                     for c in ["wrps_offense_percent","wrps_defense_percent","wrps_overall_percent"])
    if need_proxy:
        for col in ["_ppa_tot","_ppa_off","_ppa_def"]:
            if col not in df.columns:
                df[col] = None
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
    rows = [{"team": x.team, "talent": float(getattr(x, "talent", 0) or 0)} for x in items or []]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"team": [], "talent_score_0_100": []})
    mn, mx = df["talent"].min(), df["talent"].max()
    if mx == mn:
        df["talent_score_0_100"] = 50.0
    else:
        df["talent_score_0_100"] = ((df["talent"] - mn) / (mx - mn) * 100.0).round(1)
    return df[["team","talent_score_0_100"]]

def df_prev_sos_rank(year_prior: int) -> pd.DataFrame:
    try:
        srs = ratings_api.get_srs(year=year_prior)
    except Exception as e:
        print(f"[warn] srs fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})
    srs_map = {x.team: float(x.rating or 0) for x in srs}
    try:
        games = games_api.get_games(year=year_prior, season_type="both")
    except Exception as e:
        print(f"[warn] games fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})
    from collections import defaultdict
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

def df_portal(year: int) -> pd.DataFrame:
    try:
        portal = players_api.get_transfer_portal(year=year)
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
            val = float(rating) if isinstance(rating, (int,float)) else (float(stars) if isinstance(stars,(int,float)) else 1.0)
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
    def scale(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum()==0: return pd.Series([50.0]*len(s))
        mn, mx = s.min(), s.max()
        if mx == mn: return pd.Series([50.0]*len(s))
        return (s - mn) / (mx - mn) * 100.0
    df["portal_net_0_100"] = (0.5*scale(df["portal_net_count"]) + 0.5*scale(df["portal_net_value"])).round(1)
    return df[["team","portal_net_0_100","portal_net_count","portal_net_value"]]

# ---------------- Team table build ----------------
def build_team_inputs() -> pd.DataFrame:
    fbs_df, conf_map = df_fbs_teams(YEAR)
    conferences = sorted(set(fbs_df["conference"].dropna()))
    rp = df_returning(YEAR, conferences, conf_map)
    talent = df_talent(YEAR)
    sos_rank = df_prev_sos_rank(PRIOR)
    portal = df_portal(YEAR)

    df = fbs_df.merge(rp, on=["team","conference"], how="left") \
               .merge(talent, on="team", how="left") \
               .merge(portal, on="team", how="left") \
               .merge(sos_rank, on="team", how="left")

    # SOS to 0–100 (higher = tougher)
    if "prev_season_sos_rank_1_133" in df.columns:
        s = pd.to_numeric(df["prev_season_sos_rank_1_133"], errors="coerce")
        df["sos_0_100"] = ((133 - s + 1) / 133.0 * 100.0).round(1)

    # Fill missing with 50 baselines
    for col in ["wrps_offense_percent","wrps_defense_percent","wrps_overall_percent","wrps_percent_0_100","talent_score_0_100","portal_net_0_100","sos_0_100"]:
        if col not in df.columns:
            df[col] = 50.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(50.0)

    # Power (0..100)
    df["team_power_0_100"] = (
        0.40*df["wrps_offense_percent"] +
        0.25*df["wrps_defense_percent"] +
        0.20*df["talent_score_0_100"] +
        0.10*df["portal_net_0_100"] +
        0.05*df["sos_0_100"]
    ).round(1)

    # Advantage score (identical scale as power here, could be tuned later)
    df["adv_score"] = df["team_power_0_100"].astype(float)

    # Rating via z-score (clipped) → map ~10 points per sigma
    z = _zscore_clip(df["team_power_0_100"].astype(float).values, clip=2.5)
    df["team_rating"] = (z * 10.0).round(2)  # neutral field points relative to mean team

    # Sort nicely
    df.sort_values(["conference","team"], inplace=True, ignore_index=True)

    # Output team inputs
    out_csv = os.path.join(DATA_DIR, "upa_team_inputs_datadriven_v0.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {df.shape[0]} rows.", flush=True)
    return df

# ---------------- Schedule + market + scores ----------------
def build_schedule_and_market(team_inputs: pd.DataFrame) -> pd.DataFrame:
    # All games (regular season, week 1..15) — you can extend if needed
    rows = []
    for wk in range(1, 16):
        try:
            games = games_api.get_games(year=YEAR, week=wk, season_type="regular")
        except Exception as e:
            print(f"[warn] games fetch failed for week {wk}: {e}", file=sys.stderr)
            continue
        for g in games or []:
            ht = getattr(g, "home_team", None); at = getattr(g, "away_team", None)
            if not ht or not at: continue
            # Filter FCS by team presence in team_inputs (FBS only)
            if ht not in set(team_inputs["team"]) or at not in set(team_inputs["team"]):
                continue
            rows.append({
                "game_id": getattr(g, "id", None),
                "week": wk,
                "date": str(getattr(g, "start_date", ""))[:10],
                "home_team": ht,
                "away_team": at,
                "home_points": _safe_float(getattr(g, "home_points", None)),
                "away_points": _safe_float(getattr(g, "away_points", None)),
                "neutral_site": "1" if getattr(g, "neutral_site", False) else "0"
            })

    sched = pd.DataFrame(rows)
    if sched.empty:
        # still write an empty file with headers
        out_sched = os.path.join(DATA_DIR, "cfb_schedule.csv")
        sched.to_csv(out_sched, index=False)
        print(f"[warn] No schedule rows; wrote empty {out_sched}", file=sys.stderr)
        return sched

    # Market spreads (closest line we can get)
    market_rows = []
    try:
        # Pull lines for entire season (some APIs may require week loop)
        for wk in sorted(sched["week"].unique()):
            lines = bet_api.get_lines(year=YEAR, week=wk, season_type="regular")
            for ln in lines or []:
                try:
                    gid = getattr(ln, "game_id", None)
                    market = getattr(ln, "spread", None)
                    # Normalize to home-positive first (CFBD spread is usually away@home, away spread)
                    # If API returns home spread already, you may need to flip; we try to infer:
                    # Use provided home/away names to align direction.
                    if market is None:
                        continue
                    # CFBD's Lines model often gives spread from the perspective of the away team.
                    # We'll convert to HOME-POSITIVE: home = -away_spread
                    market_home = -float(market)
                    market_rows.append({"game_id": gid, "market_spread_home": market_home})
                except Exception:
                    continue
    except Exception as e:
        print(f"[warn] market lines fetch failed: {e}", file=sys.stderr)

    market_df = pd.DataFrame(market_rows).dropna().drop_duplicates(subset=["game_id"], keep="last")
    sched = sched.merge(market_df, on="game_id", how="left")

    # Add conferences
    conf_map = {r.team: r.conference for r in team_inputs[["team","conference"]].itertuples(index=False)}
    sched["home_conf"] = sched["home_team"].map(conf_map)
    sched["away_conf"] = sched["away_team"].map(conf_map)

    # Write schedule CSV
    out_sched = os.path.join(DATA_DIR, "cfb_schedule.csv")
    sched.to_csv(out_sched, index=False)
    print(f"Wrote {out_sched} with {sched.shape[0]} rows.", flush=True)
    return sched

# ---------------- Predictions build (both pipelines) ----------------
def build_predictions(team_inputs: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    # Join ratings/advantage
    base = sched.merge(
        team_inputs[["team","team_rating","adv_score"]].rename(columns={"team":"home_team","team_rating":"home_rating","adv_score":"home_adv_score"}),
        on="home_team", how="left"
    ).merge(
        team_inputs[["team","team_rating","adv_score"]].rename(columns={"team":"away_team","team_rating":"away_rating","adv_score":"away_adv_score"}),
        on="away_team", how="left"
    )

    # Handicap pipeline (home-positive raw)
    base["model_spread_home"] = (base["home_rating"] + HFA_POINTS) - base["away_rating"]

    # Advantage pipeline: gap (home-positive)
    base["adv_gap"] = (base["home_adv_score"] - base["away_adv_score"] + HFA_POINTS)

    # Calibrate handicap to market (use only rows with market)
    mask_mkt = base["market_spread_home"].notna() & base["model_spread_home"].notna()
    if mask_mkt.any():
        a1, b1 = _lin_reg(base.loc[mask_mkt, "model_spread_home"].values,
                          base.loc[mask_mkt, "market_spread_home"].values)
        a1 = max(ALPHA_CLAMP[0], min(a1, ALPHA_CLAMP[1]))
    else:
        a1, b1 = 1.0, 0.0
    base["model_spread_cal"] = (a1 * base["model_spread_home"] + b1)

    # Advantage→market regression
    mask_adv = base["market_spread_home"].notna() & base["adv_gap"].notna()
    if mask_adv.any():
        a2, b2 = _lin_reg(base.loc[mask_adv, "adv_gap"].values,
                          base.loc[mask_adv, "market_spread_home"].values)
        a2 = max(ALPHA2_CLAMP[0], min(a2, ALPHA2_CLAMP[1]))
    else:
        a2, b2 = 1.0, 0.0
    base["expected_market_spread"] = (a2 * base["adv_gap"] + b2)

    # --- Convert to book-style (home favorite negative) ---
    base["market_spread_book"] = -base["market_spread_home"]
    base["model_spread_book"] = -base["model_spread_cal"]
    base["expected_market_spread_book"] = -base["expected_market_spread"]

    # Edges (book-style & home-positive)
    base["edge_points_homepos"] = base["model_spread_cal"] - base["market_spread_home"]
    base["edge_points_book"] = base["model_spread_book"] - base["market_spread_book"]

    # Value metric (alignment: market - expected)
    base["value_points_homepos"] = base["market_spread_home"] - base["expected_market_spread"]
    base["value_points_book"] = base["market_spread_book"] - base["expected_market_spread_book"]

    # --- Outcomes for played games (Week 1 or any completed games) ---
    # We consider "played" if both points are finite numbers.
    base["played"] = np.isfinite(base["home_points"]) & np.isfinite(base["away_points"])
    base["actual_home_margin"] = base["home_points"] - base["away_points"]

    # Model pick (book-style): model_book < 0 => home pick; >0 => away pick
    def _pick_from_book(x):
        if not np.isfinite(x):
            return None
        return "HOME" if x < 0 else ("AWAY" if x > 0 else "PUSH")

    base["model_pick"] = base["model_spread_book"].apply(_pick_from_book)
    base["expected_pick"] = base["expected_market_spread_book"].apply(_pick_from_book)

    # Actual winner: HOME if margin > 0, AWAY if < 0, PUSH if == 0
    def _winner_from_margin(m):
        if not np.isfinite(m):
            return None
        if m > 0: return "HOME"
        if m < 0: return "AWAY"
        return "PUSH"

    base["actual_winner"] = base["actual_home_margin"].apply(_winner_from_margin)

    # Correctness flags (only when played and no PUSH)
    def _correct(pred, actual):
        if pred is None or actual is None: return None
        if pred == "PUSH" or actual == "PUSH": return None
        return "CORRECT" if pred == actual else "INCORRECT"

    base["model_result"] = [ _correct(p,a) for p,a in zip(base["model_pick"], base["actual_winner"]) ]
    base["expected_result"] = [ _correct(p,a) for p,a in zip(base["expected_pick"], base["actual_winner"]) ]

    # Tidy numeric formatting
    for c in ["model_spread_home","model_spread_cal","market_spread_home","expected_market_spread",
              "model_spread_book","market_spread_book","expected_market_spread_book",
              "edge_points_homepos","edge_points_book","value_points_homepos","value_points_book",
              "adv_gap","home_rating","away_rating","home_adv_score","away_adv_score"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").round(2)

    # Keep nice column order
    cols = [
        "game_id","week","date","home_team","away_team","home_conf","away_conf","neutral_site",
        "home_points","away_points","played","actual_home_margin","actual_winner",
        "home_adv_score","away_adv_score","adv_gap",
        "home_rating","away_rating",
        "market_spread_home","model_spread_home","model_spread_cal","expected_market_spread",
        "market_spread_book","model_spread_book","expected_market_spread_book",
        "edge_points_homepos","edge_points_book",
        "value_points_homepos","value_points_book",
        "model_pick","model_result","expected_pick","expected_result"
    ]
    for c in cols:
        if c not in base.columns:
            base[c] = np.nan

    preds = base[cols].copy()

    # Live edge report (top edges by absolute value in book-style)
    live_edge = preds.dropna(subset=["edge_points_book"]).copy()
    live_edge["abs_edge"] = live_edge["edge_points_book"].abs()
    live_edge.sort_values(["week","abs_edge"], ascending=[True, False], inplace=True)
    live_edge_out = live_edge.drop(columns=["abs_edge"])

    # Write outputs
    preds_out = os.path.join(DATA_DIR, "upa_predictions.csv")
    preds.to_csv(preds_out, index=False)
    print(f"Wrote {preds_out} with {preds.shape[0]} rows.", flush=True)

    live_out = os.path.join(DATA_DIR, "live_edge_report.csv")
    live_edge_out.to_csv(live_out, index=False)
    print(f"Wrote {live_out} with {live_edge_out.shape[0]} rows.", flush=True)

    return preds

# ---------------- Main ----------------
def main():
    start = datetime.now(timezone.utc)

    teams_df = build_team_inputs()
    sched_df = build_schedule_and_market(teams_df)
    preds_df = build_predictions(teams_df, sched_df)

    status = {
        "generated_at_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": YEAR,
        "teams": int(teams_df.shape[0]),
        "games": int(sched_df.shape[0]),
        "pred_rows": int(preds_df.shape[0]),
        "next_run_eta_utc": (start + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(os.path.join(DATA_DIR, "status.json"), "w") as f:
        json.dump(status, f, indent=2)

    print("Collector completed.", flush=True)

if __name__ == "__main__":
    main()