#!/usr/bin/env python
"""Run the full data build cycle for the deploy workflow.

This script mirrors the logic previously embedded directly in the GitHub Actions
workflow, making it easier to invoke multiple times (before and after model
training).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta

import pandas as pd

from agents.collect import (
    CfbdClients,
    ApiCache,
    DATA_DIR,
    write_dataset,
    load_schedule_for_year,
    discover_current_week,
    get_market_lines_for_current_week,
    build_team_inputs_datadriven,
    build_predictions_for_year,
    build_live_edge_report,
    update_live_scores,
)
from agents.storage import read_dataset, write_json_blob, read_json_blob


def _spread_count(df: pd.DataFrame) -> int:
    for col in ("spread", "market_spread_book"):
        if col in df.columns:
            try:
                return int(pd.to_numeric(df.get(col), errors="coerce").notna().sum())
            except Exception:
                return 0
    return 0


def build_site_data(*, year: int, bearer_token: str | None = None) -> None:
    data_dir = os.environ.get("DATA_DIR", DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)

    apis = CfbdClients(bearer_token=bearer_token or "")
    cache = ApiCache()

    # 1. Team inputs
    team_inputs = build_team_inputs_datadriven(year, apis, cache)
    write_dataset(team_inputs, "upa_team_inputs_datadriven_v0")
    print(f"[OK] wrote upa_team_inputs_datadriven_v0 rows={len(team_inputs)}")

    # 2. Schedule
    schedule = load_schedule_for_year(year, apis, cache)
    write_dataset(schedule, "cfb_schedule")
    print(f"[OK] wrote schedule rows={len(schedule)}")

    # 3. Markets (FanDuel primary, CFBD fallback)
    current_week = discover_current_week(schedule) or int(
        pd.to_numeric(schedule.get("week"), errors="coerce").max() or 1
    )
    markets = get_market_lines_for_current_week(year, current_week, schedule, apis, cache)
    write_dataset(markets, "market_debug")
    print(
        f"[OK] wrote market_debug rows={len(markets)} (week <= {current_week}; spreads={_spread_count(markets)})"
    )
    if read_dataset("market_unmatched").empty:
        write_dataset(
            pd.DataFrame(
                columns=[
                    "date",
                    "home_name",
                    "away_name",
                    "reason",
                    "h_best",
                    "h_score",
                    "a_best",
                    "a_score",
                ]
            ),
            "market_unmatched",
        )

    # 4. Live scores snapshot + Predictions
    live_scores = update_live_scores(year, days=3)

    preds = build_predictions_for_year(
        year,
        schedule,
        apis=apis,
        cache=cache,
        markets_df=markets,
        team_inputs_df=team_inputs,
        scoreboard_df=live_scores,
    )
    write_dataset(preds, "upa_predictions")
    print(f"[OK] wrote predictions rows={len(preds)}")

    # 6. Live edge report
    live_edge = build_live_edge_report(year, preds_df=preds)
    write_dataset(live_edge, "live_edge_report")
    print(f"[OK] wrote live_edge_report rows={len(live_edge)}")

    # 7. Status metadata
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        teams = read_dataset("upa_team_inputs_datadriven_v0")
        games = read_dataset("cfb_schedule")
        status_pred = read_dataset("upa_predictions")
        status_payload = read_json_blob("status") or {}
        status_payload.update(
            {
                "generated_at_utc": now,
                "year": year,
                "teams": int(len(teams)) if not teams.empty else 0,
                "games": int(len(games)) if not games.empty else 0,
                "pred_rows": int(len(status_pred)) if not status_pred.empty else 0,
                "next_run_eta_utc": now,
            }
        )
        write_json_blob("status", status_payload)
        with open(os.path.join(data_dir, "status.json"), "w", encoding="utf-8") as fh:
            json.dump(status_payload, fh, indent=2)
        print("[OK] updated status metadata")
    except Exception as exc:
        print(f"[warn] status update failed: {exc}")

    # Backfill summary for Status page link
    try:
        sched_df = read_dataset("cfb_schedule")
        preds_df = read_dataset("upa_predictions")
        summary = {
            "dataset": "upa_predictions",
            "predictions_rows": int(len(preds_df)),
            "predictions_rows_with_market": int(
                pd.to_numeric(preds_df.get("market_spread_book"), errors="coerce").notna().sum()
            )
            if "market_spread_book" in preds_df.columns
            else 0,
            "schedule_rows": int(len(sched_df)),
            "schedule_rows_with_market": int(
                pd.to_numeric(sched_df.get("market_spread_book"), errors="coerce").notna().sum()
            )
            if "market_spread_book" in sched_df.columns
            else 0,
        }
        write_json_blob("market_predictions_backfill", summary)
        with open(os.path.join(data_dir, "market_predictions_backfill.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("[OK] wrote market_predictions_backfill summary")
    except Exception as exc:
        print(f"[warn] unable to write backfill summary: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build site data artifacts")
    parser.add_argument("--year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bearer_token = os.environ.get("CFBD_BEARER_TOKEN", "")
    build_site_data(year=args.year, bearer_token=bearer_token)


if __name__ == "__main__":
    main()
