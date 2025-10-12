from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from agents.collect import (
    CfbdClients,
    ApiCache,
    load_schedule_for_year,
    discover_current_week,
    get_market_lines_for_current_week,
    update_live_scores,
)


def main() -> None:
    year = int(os.environ.get("YEAR", datetime.utcnow().year))
    bearer = os.environ.get("CFBD_BEARER_TOKEN", "").strip()

    apis = CfbdClients(bearer_token=bearer)
    cache = ApiCache()

    sched = load_schedule_for_year(year, apis, cache)
    wk = discover_current_week(sched)
    if not wk:
        try:
            wk = int(pd.to_numeric(sched.get("week"), errors="coerce").max() or 1)
        except Exception:
            wk = 1

    # Refresh FanDuel markets (primary) + fallback handling.
    get_market_lines_for_current_week(year, int(wk), sched, apis, cache)

    # Refresh live scoreboard snapshot.
    update_live_scores(year, days=3)


if __name__ == "__main__":
    main()
