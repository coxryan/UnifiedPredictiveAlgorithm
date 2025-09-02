#!/usr/bin/env python3
"""
UPA-F collector — end-to-end data builder

Outputs (in ./data):
- cfb_schedule.csv                    (game_id, week, date, home_team, away_team, neutral_site, market_spread)
- upa_team_inputs_datadriven_v0.csv   (team-level features incl. WRPS, Talent, Portal, SOS, power indices)
- upa_predictions.csv                 (per-game model vs market, calibrated)
- live_edge_report.csv                (top edges)
- status.json                         (dashboard heartbeat)

Env:
- BEARER_TOKEN (required)             CFBD API bearer
- UPA_YEAR (optional, default 2025)   season year
- UPA_ALLOW_HTTP_FALLBACK (0/1)       try HTTP REST schedule if SDK returns 0 rows
- SHEET_ID (optional)                 Google Sheet to upsert "Team Inputs"
- GOOGLE_APPLICATION_CREDENTIALS      path to service account JSON (for Sheets)
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Tuple

import pandas as pd

# -----------------------------
# CFBD client setup
# -----------------------------
try:
    import cfbd
except Exception:
    print("ERROR: cfbd package not available. Ensure Actions installed 'cfbd'.", file=sys.stderr)
    raise

BEARER = os.environ.get("BEARER_TOKEN", "").strip()
if not BEARER:
    print("ERROR: Missing BEARER_TOKEN secret for CollegeFootballData API.", file=sys.stderr)
    sys.exit(1)

YEAR = int(os.environ.get("UPA_YEAR", "2025"))
PRIOR = YEAR - 1
ALLOW_HTTP_FALLBACK = os.environ.get("UPA_ALLOW_HTTP_FALLBACK", "0").strip().lower() in ("1", "true", "yes")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

configuration = cfbd.Configuration(access_token=BEARER)
api_client = cfbd.ApiClient(configuration)

teams_api = cfbd.TeamsApi(api_client)
players_api = cfbd.PlayersApi(api_client)
ratings_api = cfbd.RatingsApi(api_client)
games_api = cfbd.GamesApi(api_client)
betting_api = cfbd.BettingApi(api_client)

EXPECTED_RETURNING_COLS = [
    "team", "conference",
    "wrps_offense_percent", "wrps_defense_percent", "wrps_overall_percent",
    "wrps_percent_0_100"
]

POWER5 = {"ACC", "Big Ten", "Big 12", "Pac-12", "SEC"}  # institutional factor


# -----------------------------
# Returning Production
# -----------------------------
def df_from_returning() -> pd.DataFrame:
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
                # PPA fallbacks when percent fields missing
                "_ppa_tot": getattr(it, "total_ppa", None),
                "_ppa_off": getattr(it, "total_offense_ppa", None) or (
                    getattr(it, "total_passing_ppa", 0) + getattr(it, "total_rushing_ppa", 0)
                ),
                "_ppa_def": getattr(it, "total_defense_ppa", None)
                            or getattr(it, "total_defensive_ppa", None)
                            or getattr(it, "defense_ppa", None)
                            or getattr(it, "defensive_ppa", None),
            }
            rows.append(rec)

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])

    if df.empty:
        print("[warn] returning production endpoint returned 0; using neutral 50s.", flush=True)
        teams = sorted(team_conf.keys())
        df = pd.DataFrame({
            "team": teams,
            "conference": [team_conf.get(t, "FBS") for t in teams],
            "_overall": 50.0, "_offense": 50.0, "_defense": 50.0
        })

    # normalize 0..1 → 0..100
    def normalize_percent(x):
        if pd.isna(x):
            return None
        try:
            x = float(x)
        except Exception:
            return None
        return x * 100.0 if x <= 1.0 else x

    for src, out in [
        ("_offense", "wrps_offense_percent"),
        ("_defense", "wrps_defense_percent"),
        ("_overall", "wrps_overall_percent"),
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
def df_from_talent() -> pd.DataFrame:
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
def df_prev_season_sos_rank() -> pd.DataFrame:
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
def df_transfer_portal() -> pd.DataFrame:
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
# Schedule builder (SDK week-by-week, exclude all FCS)
# -----------------------------
def build_schedule_df() -> pd.DataFrame:
    """
    Build schedule using GamesApi.get_games week-by-week with retries.
    season_type='regular', fallback 'both'. **Filters to FBS-vs-FBS only.**
    Returns: columns game_id, week, date, home_team, away_team, neutral_site, market_spread
    """
    print("BEGIN schedule: SDK week-by-week get_games()", flush=True)

    # FBS set for filtering (exclude FCS)
    try:
        fbs_set = {t.school for t in teams_api.get_fbs_teams(year=YEAR)}
    except Exception:
        fbs_set = set()

    def safe_date(g) -> str:
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
            # EXCLUDE any game with a non-FBS participant
            if fbs_set and (ht not in fbs_set or at not in fbs_set):
                continue
            rows.append({
                "game_id": getattr(g, "id", None) or getattr(g, "game_id", None),
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
        for wk in range(0, 22):  # Week 0..21
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

    cols = ["game_id", "week", "date", "home_team", "away_team", "neutral_site", "market_spread"]

    if total == 0 and ALLOW_HTTP_FALLBACK:
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
                        if not ht or not at:
                            continue
                        if fbs_set and (ht not in fbs_set or at not in fbs_set):
                            continue
                        http_rows.append({
                            "game_id": g.get("id") or g.get("game_id"),
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
        try:
            prior = games_api.get_games(year=YEAR - 1, week=1, season_type="regular") or []
            print(f"DEBUG token sanity: prior season week 1 via SDK returned {len(prior)} games", flush=True)
        except Exception as e:
            print(f"[warn] prior-season sanity check failed: {e}", file=sys.stderr)
        return pd.DataFrame(columns=cols)

    df = df[cols]
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["neutral_site"] = pd.to_numeric(df["neutral_site"], errors="coerce").fillna(0).astype(int)
    return df


# -----------------------------
# Betting lines → market spread (home positive), with safe merges
# -----------------------------
def enrich_schedule_with_market_spread(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Populate sc['market_spread'] using CFBD BettingApi.get_lines.
    Positive = home favored by X points.
    Merge primarily by game_id; fallback to (week, home_team, away_team).
    """
    if sc is None or sc.empty:
        return sc

    weeks = sorted([int(w) for w in sc["week"].dropna().unique().tolist()])
    if not weeks:
        out = sc.copy()
        if "market_spread" not in out.columns:
            out["market_spread"] = pd.NA
        return out

    line_rows = []
    for wk in weeks:
        try:
            items = betting_api.get_lines(year=YEAR, week=wk, season_type="regular") or []
            if not items:
                items = betting_api.get_lines(year=YEAR, week=wk, season_type="both") or []
        except Exception as e:
            print(f"[warn] betting lines fetch failed for week {wk}: {e}", file=sys.stderr)
            items = []

        for gi in items or []:
            gid = getattr(gi, "id", None) or getattr(gi, "game_id", None)
            ht = getattr(gi, "home_team", None)
            at = getattr(gi, "away_team", None)
            books = getattr(gi, "lines", None) or []

            spreads = []
            for book in books:
                val = None
                for attr in ("home_spread", "spread"):
                    if hasattr(book, attr):
                        try:
                            val = float(getattr(book, attr))
                            break
                        except Exception:
                            pass
                if val is None and hasattr(book, "away_spread"):
                    try:
                        away_val = float(getattr(book, "away_spread"))
                        val = -away_val
                    except Exception:
                        pass
                if val is not None:
                    spreads.append(-val)  # normalize: home favored → POSITIVE

            if spreads:
                import statistics as st
                consensus = st.median(spreads)
                line_rows.append({
                    "game_id": gid,
                    "week": wk,
                    "home_team": ht,
                    "away_team": at,
                    "market_spread": round(float(consensus), 2),
                })

    out = sc.copy()
    if not line_rows:
        print("[warn] no betting lines found; leaving market_spread blank.", file=sys.stderr)
        if "market_spread" not in out.columns:
            out["market_spread"] = pd.NA
        return out

    df_lines = pd.DataFrame(line_rows)

    # Primary: join by game_id
    out = out.drop(columns=["market_spread"], errors="ignore")
    out = out.merge(df_lines[["game_id", "market_spread"]], on="game_id", how="left")
    if "market_spread" not in out.columns:
        out["market_spread"] = pd.NA

    # Fallback: (week, home_team, away_team)
    need = out["market_spread"].isna()
    if need.any():
        join_cols = ["week", "home_team", "away_team"]
        fallback = df_lines[join_cols + ["market_spread"]].rename(columns={"market_spread": "market_spread_fb"})
        out = out.merge(fallback, on=join_cols, how="left")
        out.loc[need, "market_spread"] = out.loc[need, "market_spread_fb"]
        out.drop(columns=["market_spread_fb"], inplace=True, errors="ignore")

    return out


# -----------------------------
# Helpers for team power & calibration
# -----------------------------
def qb_boost_series(team_df: pd.DataFrame) -> pd.Series:
    """QB-leaning offense boost placeholder (replace with explicit returning-QB when available)."""
    t = pd.to_numeric(team_df.get("talent_score_0_100", 50), errors="coerce").fillna(50)
    flag = (t >= t.quantile(0.75)).astype(int)
    off = pd.to_numeric(team_df.get("wrps_offense_percent", 50), errors="coerce").fillna(50)
    boosted = off + flag * 5.0
    return boosted.clip(0, 100)


def calibrate_spread(preds_df: pd.DataFrame) -> Tuple[float, float]:
    """Learn market calibration: market ≈ α * model + β (OLS)."""
    df = preds_df.dropna(subset=["model_spread_home", "market_spread_home"]).copy()
    if df.shape[0] < 20:
        return 1.25, 0.0
    import numpy as np
    x = pd.to_numeric(df["model_spread_home"], errors="coerce")
    y = pd.to_numeric(df["market_spread_home"], errors="coerce")
    m = pd.concat([x, y], axis=1).dropna()
    if m.empty:
        return 1.25, 0.0
    X = m.iloc[:, 0].values
    Y = m.iloc[:, 1].values
    Xc, Yc = X - X.mean(), Y - Y.mean()
    denom = float((Xc ** 2).sum())
    alpha = float((Xc * Yc).sum() / denom) if denom != 0 else 1.25
    beta = float(Y.mean() - alpha * X.mean())
    return alpha, beta


# -----------------------------
# Predictions / Live Edge (FBS-only)
# -----------------------------
def build_predictions(team_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    base = team_df.copy()

    # SOS to 0..100 where higher is tougher
    if "prev_season_sos_rank_1_133" in base.columns and \
       pd.to_numeric(base["prev_season_sos_rank_1_133"], errors="coerce").notna().any():
        sosr = pd.to_numeric(base["prev_season_sos_rank_1_133"], errors="coerce")
        base["sos_0_100"] = (1 - (sosr - 1) / (sosr.max() - 1 if sosr.max() > 1 else 1)) * 100.0
    else:
        base["sos_0_100"] = 50.0

    # Institutional factor (Power 5 bump)
    base["is_p5"] = base.get("conference", "").isin(POWER5).astype(int)

    # QB-leaning offense boost
    off_wrps = qb_boost_series(base)

    # Power index
    base["team_power_0_100"] = (
        0.40 * off_wrps +
        0.25 * pd.to_numeric(base.get("wrps_defense_percent", 50), errors="coerce").fillna(50) +
        0.20 * pd.to_numeric(base.get("talent_score_0_100", 50), errors="coerce").fillna(50) +
        0.10 * pd.to_numeric(base.get("portal_net_0_100", 50), errors="coerce").fillna(50) +
        0.05  * pd.to_numeric(base.get("sos_0_100", 50), errors="coerce").fillna(50) +
        0.05  * base["is_p5"] * 100.0 / 10.0
    )

    # Rating centered; calibration will scale to points
    base["team_rating"] = base["team_power_0_100"] - 50.0

    rating = base.set_index("team")["team_rating"].to_dict()
    confmap = base.set_index("team")["conference"].to_dict()
    rated = set(base["team"].tolist())  # FBS-only list

    HFA = 2.0
    out_rows = []
    for _, r in schedule_df.iterrows():
        ht, at = r.get("home_team"), r.get("away_team")
        if pd.isna(ht) or pd.isna(at):
            continue
        # EXCLUDE any game where either team isn't in our FBS team table
        if ht not in rated or at not in rated:
            continue

        hr = float(rating.get(ht, 0.0))
        ar = float(rating.get(at, 0.0))
        neutral = bool(int(r.get("neutral_site", 0))) if str(r.get("neutral_site", "")).strip() != "" else False
        hfa = 0.0 if neutral else HFA
        model_spread = (hr + hfa) - ar  # home positive

        market = pd.to_numeric(r.get("market_spread", None), errors="coerce")
        edge_raw = model_spread - market if pd.notna(market) else None

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
            "edge_points": round(edge_raw, 2) if edge_raw is not None else None
        })

    preds = pd.DataFrame(out_rows)

    # Calibrate model → market scale (and clamp α early)
    alpha, beta = calibrate_spread(preds)
    alpha = max(0.8, min(alpha, 2.5))
    preds["model_spread_cal"] = (alpha * preds["model_spread_home"] + beta).round(2)
    preds["edge_points"] = (preds["model_spread_cal"] - preds["market_spread_home"]).round(2)
    preds["cal_alpha"] = round(alpha, 4)
    preds["cal_beta"] = round(beta, 4)

    return preds


# -----------------------------
# Main
# -----------------------------
def main():
    start = datetime.now(timezone.utc)

    # Team modules
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

    # Build schedule and enrich with market spreads
    sched_path = os.path.join(DATA_DIR, "cfb_schedule.csv")
    schedule_df = build_schedule_df()
    schedule_df = enrich_schedule_with_market_spread(schedule_df)

    sched_cols = ["game_id", "week", "date", "home_team", "away_team", "neutral_site", "market_spread"]
    if schedule_df is None or schedule_df.empty:
        pd.DataFrame(columns=sched_cols).to_csv(sched_path, index=False)
        print(f"[warn] wrote header-only schedule CSV to {sched_path} (0 rows)", file=sys.stderr)
        print("[warn] no schedule available; skipping predictions/edge.", file=sys.stderr)
        print(f"Wrote {inputs_csv} with {team_df.shape[0]} rows.", flush=True)
        return
    else:
        schedule_df.to_csv(sched_path, index=False)
        print(f"DEBUG wrote schedule CSV to {sched_path} with {len(schedule_df)} rows", flush=True)

    # Predictions + live edge (calibrated) — FBS-only
    preds_df = build_predictions(team_df, schedule_df)
    preds_csv = os.path.join(DATA_DIR, "upa_predictions.csv")
    preds_df.to_csv(preds_csv, index=False)
    print(f"DEBUG wrote predictions CSV to {preds_csv} with {preds_df.shape[0]} rows", flush=True)

    edge_csv = os.path.join(DATA_DIR, "live_edge_report.csv")
    cols = ["week", "date", "home_team", "away_team", "model_spread_cal", "market_spread_home", "edge_points"]
    preds_df.dropna(subset=["edge_points"]).reindex(columns=cols).sort_values(
        by="edge_points", key=lambda s: s.abs(), ascending=False
    ).head(200).to_csv(edge_csv, index=False)
    print(f"DEBUG wrote live edge CSV to {edge_csv}", flush=True)

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