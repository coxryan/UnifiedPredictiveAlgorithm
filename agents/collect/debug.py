from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from .config import DATA_DIR, MARKET_SOURCE, _dbg
from .cfbd_clients import CfbdClients
from .cache import ApiCache
from .schedule import load_schedule_for_year, discover_current_week
from .markets import get_market_lines_for_current_week
from .helpers import write_csv
from agents.storage.sqlite_store import read_json_blob, write_json_blob, read_table_from_path


def market_debug_entry() -> None:
    """
    Market-only pass:
      * loads schedule,
      * discovers current week,
      * fetches market lines for all weeks up to current week,
      * writes small debug artefacts into DATA_DIR:
          - market_debug.json (summary + market-source bookkeeping)
          - market_debug.csv  (resolved market lines)
    Notes:
      - Reads configuration exclusively from environment variables consumed in config.py
    """
    try:
        year = int(os.environ.get("YEAR", str(datetime.utcnow().year)))
        week_override = os.environ.get("WEEK")
        week_override = int(week_override) if (week_override or "").strip().isdigit() else None

        bearer = os.environ.get("CFBD_BEARER_TOKEN", "").strip()
        apis = CfbdClients(bearer_token=bearer)
        cache = ApiCache()

        sched = load_schedule_for_year(year, apis, cache)
        cur_week = discover_current_week(sched) or 1
        if week_override:
            cur_week = max(1, int(week_override))

        lines = get_market_lines_for_current_week(year, cur_week, sched, apis, cache)

        os.makedirs(DATA_DIR, exist_ok=True)

        debug_csv = os.path.join(DATA_DIR, "market_debug.csv")
        write_csv(lines, debug_csv)

        status_path = os.path.join(DATA_DIR, "status.json")
        status_payload = read_json_blob(status_path) or {}

        summary = {
            "year": year,
            "week_used": cur_week,
            "requested_market": (MARKET_SOURCE or "cfbd"),
            "status_market_used": status_payload.get("market_source_used"),
            "fallback_reason": status_payload.get("fallback_reason") or status_payload.get("market_fallback_reason"),
            "rows_returned": int(lines.shape[0]) if isinstance(lines, pd.DataFrame) else 0,
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        write_json_blob(os.path.join(DATA_DIR, "market_debug.json"), summary)

        _dbg(f"market_debug_entry: summary={summary}")

        # Ensure unmatched CSV exists for UI link (may be empty if no unmatched rows were produced upstream)
        try:
            unm_csv = os.path.join(DATA_DIR, "market_unmatched.csv")
            existing_df = read_table_from_path(unm_csv)
            if existing_df.empty:
                write_csv(pd.DataFrame(columns=["date","home_name","away_name","reason","h_best","h_score","a_best","a_score"]), unm_csv)
        except Exception:
            pass

    except Exception as e:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            write_json_blob(
                os.path.join(DATA_DIR, "market_debug.json"),
                {
                    "error": str(e),
                    "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "requested_market": (MARKET_SOURCE or "cfbd"),
                },
            )
        except Exception:
            pass
        raise


__all__ = ["market_debug_entry"]
