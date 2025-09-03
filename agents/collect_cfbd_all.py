#!/usr/bin/env python3
# --- make local agents/lib importable when running via "python agents/collect_cfbd_all.py"
import os, sys, json, argparse
CURR_DIR = os.path.dirname(__file__)
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

from datetime import datetime, timedelta, timezone

import pandas as pd

from lib.cache import ApiCache
from lib.cfbd_clients import build_clients
from lib.team_inputs import build_team_inputs
from lib.market import build_schedule_with_market_current_week_only
from lib.predict import build_predictions_for_year
from lib.backtest import run_backtest

def write_csv(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({df.shape[0]} rows)")

def write_status(path, year, teams, games, preds):
    start = datetime.now(timezone.utc)
    status = {
        "generated_at_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": year,
        "teams": int(teams),
        "games": int(games),
        "predictions": int(preds),
        "next_run_eta_utc": (start + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"Wrote {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=int(os.environ.get("UPA_YEAR","2025")))
    parser.add_argument("--backtest", type=int, default=int(os.environ.get("UPA_BACKTEST_YEAR","0")))
    args = parser.parse_args()

    YEAR = int(args.year)
    BACKTEST_YEAR = int(args.backtest) if int(args.backtest) and int(args.backtest) != YEAR else 0

    BEARER = os.environ.get("BEARER_TOKEN","").strip()
    if not BEARER:
        print("ERROR: Missing BEARER_TOKEN.", file=sys.stderr)
        sys.exit(1)

    DATA_DIR = "data"
    CACHE_DIR = os.environ.get("UPA_CACHE_DIR", os.path.join(DATA_DIR, ".api_cache"))
    cache = ApiCache(CACHE_DIR)

    # TTLs
    TTL = dict(
        TEAMS_FBS       = int(os.environ.get("UPA_TTL_TEAMS_FBS","86400")),
        RETURNING       = int(os.environ.get("UPA_TTL_RETURNING","604800")),
        TALENT          = int(os.environ.get("UPA_TTL_TALENT","604800")),
        PORTAL          = int(os.environ.get("UPA_TTL_PORTAL","86400")),
        SRS             = int(os.environ.get("UPA_TTL_SRS","31536000")),
        GAMES           = int(os.environ.get("UPA_TTL_GAMES","31536000")),
        LINES           = int(os.environ.get("UPA_TTL_LINES","21600")),  # 6h
        CALENDAR        = int(os.environ.get("UPA_TTL_CAL","86400"))
    )

    # Edge/value thresholds (keep parity with UI)
    EDGE_MIN  = float(os.environ.get("UPA_EDGE_MIN","2"))
    VALUE_MIN = float(os.environ.get("UPA_VALUE_MIN","1"))

    apis = build_clients(BEARER)

    # 1) Team inputs
    print(f"[live] building team inputs for {YEAR} …")
    inputs = build_team_inputs(YEAR, apis, cache,
                               ttl_fbs=TTL["TEAMS_FBS"], ttl_return=TTL["RETURNING"],
                               ttl_talent=TTL["TALENT"], ttl_portal=TTL["PORTAL"],
                               ttl_srs=TTL["SRS"], ttl_games=TTL["GAMES"])
    write_csv(os.path.join(DATA_DIR,"upa_team_inputs_datadriven_v0.csv"), inputs)

    # 2) Schedule (market for current week only)
    print(f"[live] schedule + market (current week only) for {YEAR} …")
    schedule, current_week = build_schedule_with_market_current_week_only(YEAR, apis, cache,
                                                                          ttl_games=TTL["GAMES"], ttl_lines=TTL["LINES"])
    write_csv(os.path.join(DATA_DIR,"cfb_schedule.csv"), schedule)
    print(f"[live] current_week identified as: {current_week}")

    # 3) Predictions
    print(f"[live] predictions for {YEAR} …")
    preds = build_predictions_for_year(YEAR, inputs, schedule, edge_min=EDGE_MIN, value_min=VALUE_MIN)
    write_csv(os.path.join(DATA_DIR,"upa_predictions.csv"), preds)

    live_cols = ["week","date","away_team","home_team","neutral_site",
                 "model_spread_book","market_spread_book","expected_market_spread_book",
                 "edge_points_book","value_points_book","qualified_edge_flag"]
    live_edge = preds[live_cols].copy()
    write_csv(os.path.join(DATA_DIR,"live_edge_report.csv"), live_edge)

    # 4) Status
    write_status(os.path.join(DATA_DIR,"status.json"), YEAR, inputs.shape[0], schedule.shape[0], preds.shape[0])

    # 5) Backtest (optional)
    if BACKTEST_YEAR:
        print(f"[backtest] running for {BACKTEST_YEAR} …")
        inputs_bt = build_team_inputs(BACKTEST_YEAR, apis, cache,
                                      ttl_fbs=TTL["TEAMS_FBS"], ttl_return=TTL["RETURNING"],
                                      ttl_talent=TTL["TALENT"], ttl_portal=TTL["PORTAL"],
                                      ttl_srs=TTL["SRS"], ttl_games=TTL["GAMES"])
        def build_preds_bt(year_bt, inputs_bt_local):
            sched_bt, _ = build_schedule_with_market_current_week_only(year_bt, apis, cache,
                                                                       ttl_games=TTL["GAMES"], ttl_lines=TTL["LINES"])
            return build_predictions_for_year(year_bt, inputs_bt_local, sched_bt, edge_min=EDGE_MIN, value_min=VALUE_MIN)
        run_backtest(BACKTEST_YEAR, inputs_bt, build_preds_bt, apis, cache, data_dir=DATA_DIR, ttl_games=TTL["GAMES"])

if __name__ == "__main__":
    main()