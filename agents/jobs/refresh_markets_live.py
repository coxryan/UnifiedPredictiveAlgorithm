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
)
from agents.collect.helpers import write_csv
from agents.storage.sqlite_store import write_named_table, delete_rows
from agents.fetch_live_scores import fetch_scoreboard


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
    rows = fetch_scoreboard(None)
    live_scores_columns = [
        "event_id","date","state","detail","clock","period","venue",
        "home_team","away_team","home_school","away_school","home_points","away_points"
    ]
    ls_df = pd.DataFrame(rows, columns=live_scores_columns)
    write_csv(ls_df, os.path.join(os.environ.get("DATA_DIR", "data"), "live_scores.csv"))
    if not ls_df.empty:
        store_ls = ls_df.copy()
        store_ls["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
        store_ls["season"] = year
        delete_rows("raw_espn_scoreboard", "season", year)
        write_named_table(store_ls, "raw_espn_scoreboard", if_exists="append")


if __name__ == "__main__":
    main()
