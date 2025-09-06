#!/usr/bin/env python3
# agents/collect_cfbd_all.py
# Orchestrator for UPA-F:
# - Builds team inputs
# - Builds schedule (with current week market lines)
# - Generates predictions (for all weeks)
# - Writes live edge report + status
# - Optionally runs a backtest (e.g., 2024)

import os
import json
import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd

# Package-style imports (no sys.path hacks needed)
from agents.lib.cache import ApiCache
from agents.lib.cfbd_clients import build_clients
from agents.lib.team_inputs import build_team_inputs
from agents.lib.market import build_schedule_with_market_current_week_only
from agents.lib.predict import build_predictions_for_year
from agents.lib.backtest import run_backtest

def write_csv(path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({df.shape[0]} rows)")

def write_status(path: str, year: int, teams: int, games: int, preds: int, current_week: int):
    now = datetime.now(timezone.utc)
    status = {
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "year": year,
        "current_week": current_week,
        "teams": int(teams),
        "games": int(games),
        "predictions": int(preds),
        "next_run_eta_utc": (now + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"Wrote {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=int(os.environ.get("UPA_YEAR", "2025")))
    parser.add_argument("--backtest", type=int, default=int(os.environ.get("UPA_BACKTEST_YEAR", "0")))
    args = parser.parse_args()

    YEAR = int(args.year)
    BACKTEST_YEAR = int(args.backtest) if int(args.backtest) and int(args.backtest) != YEAR else 0

    BEARER = os.environ.get("BEARER_TOKEN", "").strip()
    if not BEARER:
        raise SystemExit("ERROR: Missing BEARER_TOKEN (CFBD API)")

    EDGE_MIN = float(os.environ.get("UPA_EDGE_MIN", "2"))
    VALUE_MIN = float(os.environ.get("UPA_VALUE_MIN", "1"))

    cache = ApiCache(os.path.join("data", ".api_cache"))
    apis = build_clients(BEARER)

    # ---------------- Team inputs ----------------
    print(f"[live] building team inputs for {YEAR} …")
    inputs = build_team_inputs(YEAR, apis, cache)
    write_csv(os.path.join("data", "upa_team_inputs_datadriven_v0.csv"), inputs)
    print(f"[collector] inputs: teams={inputs.shape[0]}")

    # ---------------- Schedule ----------------
    print(f"[live] building schedule (market for current week only) for {YEAR} …")
    sched, current_week = build_schedule_with_market_current_week_only(
        YEAR, apis, cache,
        ttl_lines_nonempty=int(os.environ.get("UPA_TTL_LINES", "21600")),
        ttl_lines_empty=int(os.environ.get("UPA_TTL_LINES_EMPTY", "1800")),
    )
    write_csv(os.path.join("data", "cfb_schedule.csv"), sched)
    with_market = sched["market_spread_book"].astype(str).str.len().gt(0).sum()
    print(f"[collector] schedule: rows={sched.shape[0]}; with market={with_market}")
    print(f"[live] current_week = {current_week}")

    # ---------------- Predictions ----------------
    print(f"[live] computing predictions for {YEAR} …")
    preds = build_predictions_for_year(YEAR, inputs, sched, edge_min=EDGE_MIN, value_min=VALUE_MIN)
    write_csv(os.path.join("data", "upa_predictions.csv"), preds)
    with_market_preds = preds["market_spread_book"].notna().sum()
    print(f"[collector] predictions: rows={preds.shape[0]}; with market={with_market_preds}")

    # ---------------- Live Edge ----------------
    live_cols = [
        "week", "date", "away_team", "home_team", "neutral_site",
        "model_spread_book", "market_spread_book", "expected_market_spread_book",
        "edge_points_book", "value_points_book", "qualified_edge_flag",
    ]
    live = preds[live_cols].copy()
    write_csv(os.path.join("data", "live_edge_report.csv"), live)

    # ---------------- Status ----------------
    write_status(
        os.path.join("data", "status.json"),
        YEAR, inputs.shape[0], sched.shape[0], preds.shape[0], current_week,
    )

    # ---------------- Backtest ----------------
    if BACKTEST_YEAR:
        print(f"[backtest] running backtest for {BACKTEST_YEAR} …")
        inputs_bt = build_team_inputs(BACKTEST_YEAR, apis, cache)
        run_backtest(
            year=BACKTEST_YEAR,
            team_inputs=inputs_bt,
            apis=apis,
            cache=cache,
            data_dir="data",
        )

if __name__ == "__main__":
    main()