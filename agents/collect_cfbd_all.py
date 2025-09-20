from __future__ import annotations

"""
Compatibility shim for the legacy monolithic module.

All functionality is split across modules in `agents/collect/` and re-exported here
to keep existing imports working.
"""

import pathlib
import sys

# Allow running as ``python agents/collect_cfbd_all.py`` where relative imports lack context.
if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

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
    # odds + name resolution
    "_odds_api_fetch_fanduel","_date_from_iso","_best_fuzzy_match","_resolve_names_to_schedule",
    "_resolve_names_to_schedule_with_details","_autofix_aliases_from_unmatched","get_market_lines_fanduel_for_weeks",
    # markets
    "_cfbd_lines_to_bookstyle","get_market_lines_for_current_week",
    # debug entry
    "market_debug_entry",
]
