from __future__ import annotations

"""
Compatibility shim for the legacy monolithic module.

All functionality is split across modules in `agents/collect/` and re-exported here
to keep existing imports working.
"""

import logging
import os
import pathlib
import subprocess
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


def _configure_file_logging(data_dir: str) -> None:
    try:
        os.makedirs(data_dir, exist_ok=True)
        log_path = os.path.join(data_dir, "debug", "collector.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)
        logger.debug("File logging initialised at %s", log_path)
    except Exception:
        logger.exception("Failed to set up file logging")


def _maybe_stage_files(paths: list[str]) -> None:
    flag = os.environ.get("UPA_AUTO_GIT_ADD", "0").strip().lower()
    if flag not in {"1", "true", "yes", "y"}:
        return
    try:
        cmd = ["git", "add"] + paths
        subprocess.run(cmd, check=True)
        logger.debug("Auto-staged generated files: %s", paths)
    except subprocess.CalledProcessError as exc:
        logger.error("Auto git add failed: %s", exc)
    except FileNotFoundError:
        logger.error("git executable not found; unable to auto-stage files")

from agents.collect import (
    # config/env
    DATA_DIR,
    DATA_DB_PATH,
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
    write_dataset,
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
    update_live_scores,
    build_backtest_dataset,
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
from agents.storage import read_dataset

__all__ = [
    # config/env
    "DATA_DIR","DATA_DB_PATH","CACHE_DIR","CACHE_TTL_DAYS","CACHE_ONLY","ENABLE_SHEETS","ODDS_API_KEY","MARKET_SOURCE",
    "ODDS_CACHE_DIR","ODDS_CACHE_TTL_DAYS","REQUIRE_SCHED_MIN_ROWS","DEBUG_MARKET","MARKET_MIN_ROWS","_dbg",
    # status
    "_upsert_status_market_source",
    # core
    "ApiCache","get_odds_cache","CfbdClients","write_dataset",
    # helpers
    "_safe_float","_grade_pick_result","_apply_book_grades","_mirror_book_to_legacy_columns",
    "_normalize_percent","_scale_0_100",
    # schedule
    "discover_current_week","_dummy_schedule","_date_only","_iso_datetime_str","load_schedule_for_year",
    # teams
    "build_team_inputs_datadriven",
    "build_predictions_for_year",
    "build_live_edge_report",
    "build_backtest_dataset",
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
    p.add_argument("--backtest", type=int, default=None)
    args = p.parse_args()

    if args.market_source:
        os.environ["MARKET_SOURCE"] = str(args.market_source)
    if args.year:
        os.environ["YEAR"] = str(args.year)

    # Create market debug artifacts so UI links always resolve.
    market_debug_entry()

    # Also ensure schedule, market data, and live scores are materialized so the UI never 404s after a clean checkout.
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
            build_backtest_dataset,
            DATA_DIR,
            write_dataset,
        )
        year = int(os.environ.get("YEAR", "2025"))
        raw_token = os.environ.get("CFBD_BEARER_TOKEN", "")
        masked = f"len={len(raw_token)}" if raw_token else "len=0"
        logger.debug("collect_cfbd_all: CFBD_BEARER_TOKEN %s", masked)
        apis = CfbdClients(bearer_token=raw_token)
        logger.debug(
            "collect_cfbd_all: CFBD clients availability -> games_api=%s lines_api=%s",
            bool(apis.games_api), bool(apis.lines_api)
        )
        cache = ApiCache()
        os.makedirs(DATA_DIR, exist_ok=True)
        _configure_file_logging(DATA_DIR)
        # team inputs
        logger.debug("collect_cfbd_all: building team inputs")
        teams_df = build_team_inputs_datadriven(year, apis, cache)
        write_dataset(teams_df, "upa_team_inputs_datadriven_v0")
        logger.debug("collect_cfbd_all: wrote team inputs rows=%s", len(teams_df))
        # schedule
        logger.debug("collect_cfbd_all: loading schedule")
        sched = load_schedule_for_year(year, apis, cache)
        write_dataset(sched, "cfb_schedule")
        logger.debug("collect_cfbd_all: wrote schedule rows=%s", len(sched))
        # market CSV redundancy (if debug step failed, ensure CSV exists)
        wk = discover_current_week(sched) or 1
        logger.debug("collect_cfbd_all: fetching market lines for week=%s", wk)
        lines = get_market_lines_for_current_week(year, wk, sched, apis, cache)
        try:
            non_null_lines = int(pd.to_numeric(lines.get("spread"), errors="coerce").notna().sum()) if "spread" in lines.columns else 0
        except Exception:
            non_null_lines = 0
        logger.debug(
            "collect_cfbd_all: fetched market lines rows=%s non_null=%s",
            len(lines), non_null_lines
        )
        combined_lines = lines
        try:
            existing = read_dataset("market_debug")
            if not existing.empty:
                combined_lines = pd.concat([existing, lines], ignore_index=True)
                keep_cols = [c for c in ["game_id", "week", "home_team", "away_team"] if c in combined_lines.columns]
                if keep_cols:
                    combined_lines = combined_lines.drop_duplicates(subset=keep_cols, keep="last")
        except Exception:
            logger.exception("collect_cfbd_all: unable to merge with existing market_debug snapshot; using fresh data only")
            combined_lines = lines
        write_dataset(combined_lines, "market_debug")
        try:
            combined_non_null = int(pd.to_numeric(combined_lines.get("spread"), errors="coerce").notna().sum()) if "spread" in combined_lines.columns else 0
        except Exception:
            combined_non_null = 0
        logger.debug(
            "collect_cfbd_all: wrote market_debug rows=%s non_null=%s (new=%s)",
            len(combined_lines), combined_non_null, len(lines)
        )
        # predictions + live edge
        logger.debug("collect_cfbd_all: refreshing live scores snapshot")
        live_scores_df = update_live_scores(year, days=3)

        logger.debug("collect_cfbd_all: building predictions")
        preds = build_predictions_for_year(
            year,
            sched,
            apis=apis,
            cache=cache,
            markets_df=combined_lines,
            team_inputs_df=teams_df,
            scoreboard_df=live_scores_df,
        )
        write_dataset(preds, "upa_predictions")
        synthetic_count = int(preds.get("market_is_synthetic", pd.Series(dtype=int)).sum()) if "market_is_synthetic" in preds.columns else 0
        non_null_predictions = int(pd.to_numeric(preds.get("market_spread_book"), errors="coerce").notna().sum()) if "market_spread_book" in preds.columns else 0
        logger.debug(
            "collect_cfbd_all: wrote predictions rows=%s synthetic=%s market_non_null=%s",
            len(preds), synthetic_count, non_null_predictions
        )
        if args.backtest:
            try:
                backtest_year = int(args.backtest)
                logger.debug("collect_cfbd_all: running backtest capture for %s", backtest_year)
                build_backtest_dataset(backtest_year, apis=apis, cache=cache)
            except Exception:
                logger.exception("collect_cfbd_all: backtest capture failed")

        logger.debug("collect_cfbd_all: building live edge report")
        edge = build_live_edge_report(year, preds_df=preds)
        write_dataset(edge, "live_edge_report")
        logger.debug("collect_cfbd_all: wrote live_edge rows=%s", len(edge))

        generated_paths = [
            DATA_DB_PATH,
            os.path.join(DATA_DIR, "debug", "collector.log"),
        ]
        _maybe_stage_files(generated_paths)
    except Exception:
        logger.exception("collect_cfbd_all: data preparation failed")
