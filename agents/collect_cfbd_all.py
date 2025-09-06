"""
Entry-point script (module mode) to produce all live CSVs (and optional backtest).
Now only fetches market for the CURRENT WEEK; for all other weeks we stamp
market_spread_book=0.0 and market_is_synthetic=True so downstream logic can
differentiate. Adds clearer logging of market fetch/join.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd

from agents.lib.cache import ApiCache
from agents.lib.cfbd_clients import CfbdClients
from agents.lib.team_inputs import build_team_inputs_datadriven
from agents.lib.market import (
    current_week_from_cfbd,
    build_schedule_with_market_current_week_only,
)
from agents.lib.predict import build_predictions_and_edge
from agents.lib.backtest import run_backtest

DATA_DIR = Path("data")
CACHE_DIR = Path(".upacache")
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path, msg: str = "") -> None:
    df.to_csv(path, index=False)
    if msg:
        print(msg)


def write_status(current_week: int) -> None:
    status = {"ok": True, "current_week": int(current_week)}
    (DATA_DIR / "status.json").write_text(pd.Series(status).to_json())
    print("Wrote data/status.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--backtest", required=True, help="0 to skip, or season year (e.g., 2024)")
    args = parser.parse_args()

    season = args.year
    do_backtest = str(args.backtest).strip() != "0"

    bearer = os.environ.get("BEARER_TOKEN", "").strip()
    if not bearer:
        print("[warn] BEARER_TOKEN not set; CFBD calls that require auth may be rate-limited.")

    cache = ApiCache(root=CACHE_DIR)
    apis = CfbdClients(bearer=bearer, cache=cache)

    # 1) Team inputs
    print(f"[live] building team inputs for {season} …")
    inputs = build_team_inputs_datadriven(apis=apis, season=season)
    write_csv(
        inputs,
        DATA_DIR / "upa_team_inputs_datadriven_v0.csv",
        f"Wrote data/upa_team_inputs_datadriven_v0.csv ({len(inputs)} rows)",
    )
    print(f"[collector] inputs: teams={len(inputs)}")

    # 2) Schedule + market (current week only; others stamped with 0 + synthetic)
    cur_week = current_week_from_cfbd(apis=apis, season=season)
    print(f"[live] building schedule (market for current week only) for {season} …")
    sched = build_schedule_with_market_current_week_only(
        apis=apis,
        season=season,
        current_week=cur_week,
    )
    write_csv(sched, DATA_DIR / "cfb_schedule.csv", f"Wrote data/cfb_schedule.csv ({len(sched)} rows)")
    has_mkt = sched["market_spread_book"].notna().sum() if "market_spread_book" in sched else 0
    print(f"[collector] schedule: rows={len(sched)}; with market={has_mkt}")
    print(f"[live] current_week = {cur_week}")

    # 3) Predictions + live edge
    print(f"[live] computing predictions for {season} …")
    preds = build_predictions_and_edge(
        inputs_df=inputs,
        schedule_df=sched,
        season=season,
    )
    write_csv(preds, DATA_DIR / "upa_predictions.csv", f"Wrote data/upa_predictions.csv ({len(preds)} rows)")
    has_mkt_preds = preds.query("(market_is_synthetic == False) and market_spread_book==market_spread_book").shape[0] if "market_is_synthetic" in preds else preds["market_spread_book"].notna().sum()
    print(f"[collector] predictions: rows={len(preds)}; with real-market={has_mkt_preds}")

    live_edge = preds[
        [
            "week", "date", "away_team", "home_team", "neutral_site",
            "model_spread_book", "market_spread_book", "expected_market_spread_book",
            "edge_points_book", "value_points_book", "qualified_edge_flag", "market_is_synthetic"
        ]
    ].copy()
    write_csv(live_edge, DATA_DIR / "live_edge_report.csv", f"Wrote data/live_edge_report.csv ({len(live_edge)} rows)")

    write_status(current_week=cur_week)

    # 4) Optional backtest
    if do_backtest:
        print(f"[backtest] running for {args.backtest} …")
        bt_summary, bt_preds = run_backtest(
            season=int(args.backtest),
            inputs_live=inputs,
            apis=apis,
            cache=cache,
            data_dir=str(DATA_DIR),
        )
        write_csv(bt_preds, DATA_DIR / "backtest_predictions_2024.csv",
                  f"Wrote data/backtest_predictions_2024.csv ({len(bt_preds)} rows)")
        write_csv(bt_summary, DATA_DIR / "backtest_summary_2024.csv",
                  f"Wrote data/backtest_summary_2024.csv ({len(bt_summary)} rows)")


if __name__ == "__main__":
    main()