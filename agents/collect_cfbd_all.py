#!/usr/bin/env python3
import os, sys, argparse, json
from datetime import datetime, timedelta, timezone
import pandas as pd

# path shim
_CURR = os.path.dirname(os.path.abspath(__file__))
if _CURR not in sys.path: sys.path.insert(0, _CURR)
_ROOT = os.path.dirname(_CURR)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

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

def write_status(path, year, teams, games, preds, current_week):
    now = datetime.now(timezone.utc)
    status = {
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": year,
        "current_week": current_week,
        "teams": int(teams),
        "games": int(games),
        "predictions": int(preds),
        "next_run_eta_utc": (now + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(path, "w") as f: json.dump(status, f, indent=2)
    print(f"Wrote {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=int(os.environ.get("UPA_YEAR","2025")))
    parser.add_argument("--backtest", type=int, default=int(os.environ.get("UPA_BACKTEST_YEAR","0")))
    args = parser.parse_args()

    YEAR, BACKTEST_YEAR = args.year, args.backtest
    BEARER = os.environ.get("BEARER_TOKEN","").strip()
    if not BEARER: raise SystemExit("ERROR: Missing BEARER_TOKEN")

    EDGE_MIN=float(os.environ.get("UPA_EDGE_MIN","2"))
    VALUE_MIN=float(os.environ.get("UPA_VALUE_MIN","1"))

    cache=ApiCache(os.path.join("data",".api_cache"))
    apis=build_clients(BEARER)

    # inputs
    inputs=build_team_inputs(YEAR,apis,cache)
    write_csv("data/upa_team_inputs_datadriven_v0.csv",inputs)

    # schedule
    sched,curr_week=build_schedule_with_market_current_week_only(YEAR,apis,cache)
    write_csv("data/cfb_schedule.csv",sched)

    # predictions
    preds=build_predictions_for_year(YEAR,inputs,sched,edge_min=EDGE_MIN,value_min=VALUE_MIN)
    write_csv("data/upa_predictions.csv",preds)

    live_cols=["week","date","away_team","home_team","neutral_site",
               "model_spread_book","market_spread_book","expected_market_spread_book",
               "edge_points_book","value_points_book","qualified_edge_flag"]
    live=preds[live_cols].copy()
    write_csv("data/live_edge_report.csv",live)

    write_status("data/status.json",YEAR,inputs.shape[0],sched.shape[0],preds.shape[0],curr_week)

    if BACKTEST_YEAR and BACKTEST_YEAR!=YEAR:
        inputs_bt=build_team_inputs(BACKTEST_YEAR,apis,cache)
        run_backtest(year=BACKTEST_YEAR,team_inputs=inputs_bt,apis=apis,cache=cache,data_dir="data")

if __name__=="__main__": main()