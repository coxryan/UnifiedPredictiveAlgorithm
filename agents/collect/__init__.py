from .config import (
    DATA_DIR,
    DATA_DB_PATH,
    CACHE_DIR,
    CACHE_TTL_DAYS,
    CACHE_DB_PATH,
    CACHE_ONLY,
    ENABLE_SHEETS,
    ODDS_API_KEY,
    MARKET_SOURCE,
    FANDUEL_CACHE_ONLY,
    ODDS_CACHE_DIR,
    ODDS_CACHE_TTL_DAYS,
    REQUIRE_SCHED_MIN_ROWS,
    DEBUG_MARKET,
    MARKET_MIN_ROWS,
    _dbg,
)

from .status import _upsert_status_market_source
from .cache import ApiCache, get_odds_cache
from .cfbd_clients import CfbdClients
from .helpers import (
    write_dataset,
    _safe_float,
    _grade_pick_result,
    _apply_book_grades,
    _mirror_book_to_legacy_columns,
    _normalize_percent,
    _scale_0_100,
)
from .schedule import (
    discover_current_week,
    _dummy_schedule,
    _date_only,
    _iso_datetime_str,
    load_schedule_for_year,
)
from .team_inputs import build_team_inputs_datadriven
from .stats_cfbd import build_team_stat_features
from .predictions import build_predictions_for_year
from .live_edge import build_live_edge_report
from .live_scores import update_live_scores
from .backtest import build_backtest_dataset
from .odds_fanduel import (
    _odds_api_fetch_fanduel,
    _date_from_iso,
    _best_fuzzy_match,
    _resolve_names_to_schedule,
    _resolve_names_to_schedule_with_details,
    _autofix_aliases_from_unmatched,
    get_market_lines_fanduel_for_weeks,
)
from .markets import _cfbd_lines_to_bookstyle, get_market_lines_for_current_week
from .debug import market_debug_entry

__all__ = [
    # config
    "DATA_DIR","DATA_DB_PATH","CACHE_DIR","CACHE_TTL_DAYS","CACHE_DB_PATH","CACHE_ONLY","ENABLE_SHEETS","ODDS_API_KEY",
    "MARKET_SOURCE","FANDUEL_CACHE_ONLY","ODDS_CACHE_DIR","ODDS_CACHE_TTL_DAYS","REQUIRE_SCHED_MIN_ROWS","DEBUG_MARKET",
    "MARKET_MIN_ROWS","_dbg",
    # core utils
    "ApiCache","get_odds_cache","CfbdClients","write_dataset",
    # helpers
    "_safe_float","_grade_pick_result","_apply_book_grades","_mirror_book_to_legacy_columns",
    "_normalize_percent","_scale_0_100",
    # schedule
    "discover_current_week","_dummy_schedule","_date_only","_iso_datetime_str","load_schedule_for_year",
    # teams
    "build_team_inputs_datadriven",
    "build_team_stat_features",
    "build_predictions_for_year",
    "build_live_edge_report",
    "update_live_scores",
    "build_backtest_dataset",
    # odds/fanduel + name resolution
    "_odds_api_fetch_fanduel","_date_from_iso","_best_fuzzy_match","_resolve_names_to_schedule",
    "_resolve_names_to_schedule_with_details","_autofix_aliases_from_unmatched","get_market_lines_fanduel_for_weeks",
    # markets
    "_cfbd_lines_to_bookstyle","get_market_lines_for_current_week",
    # entry
    "market_debug_entry",
    # status
    "_upsert_status_market_source",
]
