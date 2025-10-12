#!/usr/bin/env python
"""One-off backtest data builder.

Fetches schedule/markets for a historical season, regenerates predictions using the
current model pipeline, and writes `upa_predictions_<year>_backtest` plus a summary
table into the SQLite store so the UI Backtest tab can display results.
"""

from __future__ import annotations

import argparse
import logging
import os

from agents.collect import (
    ApiCache,
    CfbdClients,
    build_backtest_dataset,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("UPA_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def build_backtest(year: int, bearer_token: str | None = None) -> None:
    _configure_logging()
    logger = logging.getLogger("build_backtest")
    token = bearer_token or os.environ.get("CFBD_BEARER_TOKEN", "")
    masked = f"len={len(token)}" if token else "len=0"
    logger.info("Starting backtest build for %s (CFBD token %s)", year, masked)

    apis = CfbdClients(bearer_token=token or "")
    cache = ApiCache()

    preds, summary = build_backtest_dataset(year, apis=apis, cache=cache)
    logger.info("Backtest build finished: predictions=%s summary=%s", len(preds), len(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backtest datasets into SQLite")
    parser.add_argument("--year", type=int, default=2024, help="Season to backtest (default: 2024)")
    parser.add_argument(
        "--cfbd-token",
        dest="cfbd_token",
        type=str,
        default=None,
        help="Override CFBD bearer token (falls back to CFBD_BEARER_TOKEN env)",
    )
    args = parser.parse_args()
    build_backtest(year=args.year, bearer_token=args.cfbd_token)


if __name__ == "__main__":
    main()
