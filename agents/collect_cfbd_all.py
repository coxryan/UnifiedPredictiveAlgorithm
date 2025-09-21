from __future__ import annotations

"""
Compatibility shim for the legacy monolithic module.

All functionality is split across modules in `agents/collect/` and re-exported here
to keep existing imports working.
"""

import logging
import os
import pathlib
import sys

# Allow running as ``python agents/collect_cfbd_all.py`` where relative imports lack context.
if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=os.environ.get("UPA_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
logger.debug("collect_cfbd_all bootstrap complete")

from agents.collect import (
    # config/env
    DATA_DIR,
    CACHE_DIR,
    CACHE_TTL_DAYS,
    CACHE_ONLY,
    ENABLE_SHEETS,
    ODDS_API_KEY,
    MARKET_SOURCE,
    ODDS_CACHE_DIR,
    ODDS_CACHE_TTL_DAYS,
    REQUIRE_SCHED_MIN_ROWS,
    DEBUG_MARKET,
    MARKET_MIN_ROWS,
    _dbg,
    # status
    _upsert_status_market_source,
    # core
    ApiCache,
    get_odds_cache,
    CfbdClients,
    write_csv,
    # helpers
    _safe_float,
    _grade_pick_result,
    _apply_book_grades,
    _mirror_book_to_legacy_columns,
    _normalize_percent,
    _scale_0_100,
    # schedule
    discover_current_week,
    _dummy_schedule,
    _date_only,
    _iso_datetime_str,
    load_schedule_for_year,
    # teams
    build_team_inputs_datadriven,
    build_predictions_for_year,
    build_live_edge_report,
    # odds + name resolution
    _odds_api_fetch_fanduel,
    _date_from_iso,
    _best_fuzzy_match,
    _resolve_names_to_schedule,
    _resolve_names_to_schedule_with_details,
    _autofix_aliases_from_unmatched,
    get_market_lines_fanduel_for_weeks,
    # markets
    _cfbd_lines_to_bookstyle,
    get_market_lines_for_current_week,
    # debug entry
    market_debug_entry,
)

__all__ = [
    # config/env
    "DATA_DIR","CACHE_DIR","CACHE_TTL_DAYS","CACHE_ONLY","ENABLE_SHEETS","ODDS_API_KEY","MARKET_SOURCE",
    "ODDS_CACHE_DIR","ODDS_CACHE_TTL_DAYS","REQUIRE_SCHED_MIN_ROWS","DEBUG_MARKET","MARKET_MIN_ROWS","_dbg",
    # status
    "_upsert_status_market_source",
    # core
    "ApiCache","get_odds_cache","CfbdClients","write_csv",
    # helpers
    "_safe_float","_grade_pick_result","_apply_book_grades","_mirror_book_to_legacy_columns",
    "_normalize_percent","_scale_0_100",
    # schedule
    "discover_current_week","_dummy_schedule","_date_only","_iso_datetime_str","load_schedule_for_year",
    # teams
    "build_team_inputs_datadriven",
    "build_predictions_for_year",
    "build_live_edge_report",
    # odds + name resolution
    "_odds_api_fetch_fanduel","_date_from_iso","_best_fuzzy_match","_resolve_names_to_schedule",
    "_resolve_names_to_schedule_with_details","_autofix_aliases_from_unmatched","get_market_lines_fanduel_for_weeks",
    # markets
    "_cfbd_lines_to_bookstyle","get_market_lines_for_current_week",
    # debug entry
    "market_debug_entry",
]


# Minimal runner to ensure debug artifacts exist for the UI.
# Accept --market-source/--year flags for compatibility; env drives behavior.
if __name__ == "__main__":
    import argparse
    from agents.collect import market_debug_entry  # re-exported above

    p = argparse.ArgumentParser()
    p.add_argument("--market-source", dest="market_source", type=str, default=os.environ.get("MARKET_SOURCE", "fanduel"))
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--backtest", type=int, default=None)  # accepted but unused here
    args = p.parse_args()

    if args.market_source:
        os.environ["MARKET_SOURCE"] = str(args.market_source)
    if args.year:
        os.environ["YEAR"] = str(args.year)

    # Create data/market_debug.json and data/market_debug.csv so UI links always resolve.
    market_debug_entry()

    # Also ensure schedule, market CSV, and live scores are materialized so the UI never 404s after a clean checkout.
    try:
        import pandas as pd
        from agents.collect import (
            CfbdClients,
            ApiCache,
            load_schedule_for_year,
            discover_current_week,
            get_market_lines_for_current_week,
            build_team_inputs_datadriven,
            build_predictions_for_year,
            build_live_edge_report,
            DATA_DIR,
            write_csv,
        )
        from agents.fetch_live_scores import fetch_scoreboard
        year = int(os.environ.get("YEAR", "2025"))
        apis = CfbdClients(bearer_token=os.environ.get("CFBD_BEARER_TOKEN",""))
        cache = ApiCache()
        os.makedirs(DATA_DIR, exist_ok=True)
        # team inputs
        teams_df = build_team_inputs_datadriven(year, apis, cache)
        write_csv(teams_df, os.path.join(DATA_DIR, "upa_team_inputs_datadriven_v0.csv"))
        logger.debug("collect_cfbd_all: wrote team inputs rows=%s", len(teams_df))
        # schedule
        sched = load_schedule_for_year(year, apis, cache)
        write_csv(sched, os.path.join(DATA_DIR, "cfb_schedule.csv"))
        logger.debug("collect_cfbd_all: wrote schedule rows=%s", len(sched))
        # market CSV redundancy (if debug step failed, ensure CSV exists)
        wk = discover_current_week(sched) or 1
        lines = get_market_lines_for_current_week(year, wk, sched, apis, cache)
        write_csv(lines, os.path.join(DATA_DIR, "market_debug.csv"))
        logger.debug("collect_cfbd_all: wrote market_debug rows=%s", len(lines))
        # predictions + live edge
        preds = build_predictions_for_year(
            year,
            sched,
            apis=apis,
            cache=cache,
            markets_df=lines,
            team_inputs_df=teams_df,
        )
        write_csv(preds, os.path.join(DATA_DIR, "upa_predictions.csv"))
        logger.debug("collect_cfbd_all: wrote predictions rows=%s synthetic=%s", len(preds), int(preds.get("market_is_synthetic", pd.Series(dtype=int)).sum() if "market_is_synthetic" in preds.columns else 0))
        edge = build_live_edge_report(year, preds_df=preds)
        write_csv(edge, os.path.join(DATA_DIR, "live_edge_report.csv"))
        logger.debug("collect_cfbd_all: wrote live_edge rows=%s", len(edge))
        # live scores
        try:
            rows = fetch_scoreboard(None)
            pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "live_scores.csv"), index=False)
        except Exception:
            pass
    except Exception:
        logger.exception("collect_cfbd_all: data preparation failed")
