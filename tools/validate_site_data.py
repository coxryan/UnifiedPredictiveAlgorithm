#!/usr/bin/env python
"""Validate core datasets for deploy workflow."""
from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from agents.storage import read_dataset


def validate_site_data(*, year: int) -> None:
    fail: list[str] = []

    # Schedule checks
    try:
        sched = read_dataset("cfb_schedule")
        if len(sched) < 200:
            fail.append(f"schedule too small: {len(sched)} rows")
        dts = pd.to_datetime(sched.get("date"), errors="coerce")
        if dts.isna().all():
            fail.append("schedule has no valid dates")
        else:
            if hasattr(dts, "dt") and getattr(dts.dt, "tz", None) is not None:
                try:
                    dts = dts.dt.tz_convert("America/Los_Angeles")
                except Exception:
                    try:
                        dts = dts.dt.tz_localize("America/Los_Angeles")
                    except Exception:
                        pass
                try:
                    dts = dts.dt.tz_localize(None)
                except Exception:
                    pass
            max_d = dts.dropna().max()
            try:
                today_pt = pd.Timestamp.now(tz="America/Los_Angeles").date()
            except Exception:
                today_pt = pd.Timestamp.utcnow().date()
            max_date = None
            if isinstance(max_d, pd.Timestamp):
                max_date = max_d.date()
            elif pd.notna(max_d):
                try:
                    max_date = pd.Timestamp(max_d).date()
                except Exception:
                    max_date = None
            if max_date and max_date < (today_pt - timedelta(days=1)):
                fail.append(f"schedule stale: max date {max_date} < (today-1)")
    except Exception as exc:
        fail.append(f"schedule load error: {exc}")

    # Predictions must exist
    try:
        preds = read_dataset("upa_predictions")
        if len(preds) == 0:
            print("::warning::predictions empty (0 rows)")
    except Exception as exc:
        fail.append(f"predictions load error: {exc}")

    # Markets must exist
    try:
        market_debug = read_dataset("market_debug")
        if len(market_debug) == 0:
            fail.append("market_debug empty (0 rows)")
    except Exception as exc:
        fail.append(f"market_debug load error: {exc}")

    # Live scores should be readable
    try:
        _ = read_dataset("live_scores")
    except Exception as exc:
        fail.append(f"live_scores load error: {exc}")

    if fail:
        print("::error::Data validation failed:")
        for item in fail:
            print(" -", item)
        raise SystemExit(1)
    print("[OK] data validation passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate site datasets")
    parser.add_argument("--year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_site_data(year=args.year)


if __name__ == "__main__":
    main()
