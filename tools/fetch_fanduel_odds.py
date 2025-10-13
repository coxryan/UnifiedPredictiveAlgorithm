#!/usr/bin/env python
"""Fetch FanDuel spreads via The Odds API and persist a debug snapshot."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import pandas as pd

from agents.collect import (
    CfbdClients,
    ApiCache,
    load_schedule_for_year,
    discover_current_week,
    get_odds_cache,
)
from agents.collect.odds_fanduel import get_market_lines_fanduel_for_weeks
from agents.storage import read_dataset, write_dataset


def _load_schedule(year: int, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    try:
        sched = read_dataset("cfb_schedule")
        if not sched.empty:
            return sched
    except Exception:
        pass
    return load_schedule_for_year(year, apis, cache)


def _resolve_weeks(schedule: pd.DataFrame) -> List[int]:
    current = discover_current_week(schedule) or int(
        pd.to_numeric(schedule.get("week"), errors="coerce").max() or 1
    )
    next_week = current + 1
    return sorted({max(1, int(current)), max(1, int(next_week))})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FanDuel spreads via The Odds API")
    parser.add_argument("--year", type=int, default=datetime.utcnow().year)
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="*",
        help="Specific weeks to fetch (defaults to current + next)",
    )
    parser.add_argument("--out", type=str, default="data/debug/fanduel_odds_snapshot.json")
    args = parser.parse_args()

    apis = CfbdClients(bearer_token=os.environ.get("CFBD_BEARER_TOKEN", ""))
    cache = ApiCache()
    schedule = _load_schedule(args.year, apis, cache)
    weeks = args.weeks if args.weeks else _resolve_weeks(schedule)

    odds_df, stats = get_market_lines_fanduel_for_weeks(
        args.year, weeks, schedule, get_odds_cache()
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    odds_df.to_json(args.out, orient="records", indent=2)
    print(
        f"[OK] wrote {len(odds_df)} FanDuel odds rows to {args.out} for weeks={weeks} (stats={json.dumps(stats)})"
    )

    # Persist a lightweight dataset for local inspection
    try:
        write_dataset(odds_df, "fanduel_odds_snapshot")
    except Exception as exc:
        print(f"[warn] unable to write dataset: {exc}")


if __name__ == "__main__":
    main()

