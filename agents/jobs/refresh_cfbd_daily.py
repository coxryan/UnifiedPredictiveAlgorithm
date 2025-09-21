from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from agents.collect import (
    CfbdClients,
    ApiCache,
    load_schedule_for_year,
    discover_current_week,
)
import agents.collect.markets as markets


def main() -> None:
    year = int(os.environ.get("YEAR", datetime.utcnow().year))
    bearer = os.environ.get("CFBD_BEARER_TOKEN", "").strip()

    # Force CFBD market usage for this run.
    markets.MARKET_SOURCE = "cfbd"

    apis = CfbdClients(bearer_token=bearer)
    cache = ApiCache()

    sched = load_schedule_for_year(year, apis, cache)
    wk = discover_current_week(sched)
    if not wk:
        try:
            wk = int(pd.to_numeric(sched.get("week"), errors="coerce").max() or 1)
        except Exception:
            wk = 1

    markets.get_market_lines_for_current_week(year, int(wk), sched, apis, cache)


if __name__ == "__main__":
    main()
