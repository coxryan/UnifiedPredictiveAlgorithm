from __future__ import annotations

from typing import Optional

import logging
import os
import numpy as np
import pandas as pd

from .cache import ApiCache
from .schedule import discover_current_week
from .team_inputs import build_team_inputs_datadriven
from .markets import get_market_lines_for_current_week
from .cfbd_clients import CfbdClients

logger = logging.getLogger(__name__)


def _sanitize_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _rating_from_team_inputs(team_inputs: pd.DataFrame) -> pd.Series:
    if team_inputs is None or team_inputs.empty:
        return pd.Series(dtype="float64")
    df = team_inputs.copy()
    for col in ["wrps_percent_0_100", "talent_score_0_100", "srs_score_0_100"]:
        if col in df.columns:
            df[col] = _sanitize_numeric(df[col])
        else:
            df[col] = 0.0
    # Weighted composite score (0-1 scale)
    w_wrps = df["wrps_percent_0_100"].fillna(50.0) / 100.0
    w_talent = df["talent_score_0_100"].fillna(50.0) / 100.0
    w_srs = df["srs_score_0_100"].fillna(50.0) / 100.0
    rating = (0.45 * w_wrps) + (0.30 * w_talent) + (0.25 * w_srs)
    rating.index = df["team"].astype(str)
    # Ensure unique index by collapsing duplicate team rows (average scores)
    rating = rating.groupby(level=0).mean()
    return rating


def _market_lookup(markets_df: pd.DataFrame) -> pd.DataFrame:
    if markets_df is None or markets_df.empty:
        return pd.DataFrame(columns=[
            "game_id",
            "week",
            "home_team",
            "away_team",
            "market_spread_book",
            "market_spread_fanduel",
            "market_spread_cfbd",
        ])
    df = markets_df.copy()
    raw_cols: set[str] = set()
    if "market_spread_book" not in df.columns:
        if "spread" in df.columns:
            df["market_spread_book_raw"] = df["spread"]
            df["market_spread_book"] = df["spread"]
        elif "point_home_book" in df.columns:
            df["market_spread_book_raw"] = df["point_home_book"]
            df["market_spread_book"] = df["point_home_book"]
    else:
        df["market_spread_book_raw"] = df["market_spread_book"]
    for src, dst in (
        ("spread_fanduel", "market_spread_fanduel"),
        ("market_spread_fanduel", "market_spread_fanduel"),
        ("spread_cfbd", "market_spread_cfbd"),
        ("market_spread_cfbd", "market_spread_cfbd"),
    ):
        if src in df.columns:
            raw_col = f"{dst}_raw"
            if raw_col not in df.columns:
                df[raw_col] = df[src]
            else:
                mask = df[src].notna()
                if mask.any():
                    df.loc[mask, raw_col] = df.loc[mask, src]
            raw_cols.add(raw_col)
            df[dst] = _sanitize_numeric(df[src])
    for col in ("game_id", "week"):
        if col in df.columns:
            df[col] = _sanitize_numeric(df[col])
    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "market_spread_source" in df.columns:
        df["market_spread_source"] = (
            df["market_spread_source"].astype(str).str.strip().str.lower()
        )
    keep_cols = [
        c
        for c in [
            "game_id",
            "week",
            "home_team",
            "away_team",
            "market_spread_book",
            "market_spread_fanduel",
            "market_spread_cfbd",
            "market_spread_source",
        ]
        if c in df.columns
    ]
    for raw_col in sorted(raw_cols | {"market_spread_book_raw"}):
        if raw_col in df.columns and raw_col not in keep_cols:
            keep_cols.append(raw_col)
    for raw_col in sorted(raw_cols):
        if raw_col in df.columns:
            keep_cols.append(raw_col)
    out = df[keep_cols].drop_duplicates()
    return out


def build_predictions_for_year(
    year: int,
    sched_df: pd.DataFrame,
    apis: Optional[CfbdClients] = None,
    cache: Optional[ApiCache] = None,
    markets_df: Optional[pd.DataFrame] = None,
    team_inputs_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return per-game prediction rows with basic edges/value signals.

    This builder intentionally uses only public CFBD inputs so it can run in CI without
    bespoke model artefacts. It emits the canonical columns consumed by the dashboard.
    """

    if sched_df is None or sched_df.empty:
        return pd.DataFrame(
            columns=[
                "week",
                "date",
                "away_team",
                "home_team",
                "neutral_site",
                "model_spread_book",
                "market_spread_book",
                "expected_market_spread_book",
                "edge_points_book",
                "value_points_book",
                "qualified_edge_flag",
                "game_id",
                "model_pick_side",
                "played",
            ]
        )

    logger.debug(
        "build_predictions_for_year:start year=%s sched_rows=%s markets_rows=%s team_inputs_rows=%s",
        year,
        len(sched_df),
        0 if markets_df is None else len(markets_df),
        0 if team_inputs_df is None else len(team_inputs_df),
    )

    sched = sched_df.copy()
    sched["week"] = _sanitize_numeric(sched["week"]) if "week" in sched.columns else 0
    for team_col in ("home_team", "away_team"):
        if team_col in sched.columns:
            sched[team_col] = sched[team_col].astype(str).str.strip()

    cache = cache or ApiCache()
    apis = apis or CfbdClients(bearer_token="")

    if team_inputs_df is None:
        team_inputs_df = build_team_inputs_datadriven(year, apis, cache)
    team_ratings = _rating_from_team_inputs(team_inputs_df)
    logger.debug("build_predictions_for_year: team ratings available for %s teams", len(team_ratings))

    if markets_df is None:
        current_week = discover_current_week(sched) or int(_sanitize_numeric(sched["week"]).max() or 1)
        markets_df = get_market_lines_for_current_week(year, int(current_week), sched, apis, cache)

    markets = _market_lookup(markets_df)
    logger.debug(
        "build_predictions_for_year: markets rows=%s unique games=%s",
        len(markets),
        markets[[c for c in ["game_id", "week"] if c in markets.columns]].drop_duplicates().shape[0]
        if not markets.empty
        else 0,
    )

    # Merge markets by game_id when available, else week + teams fallback
    preds = sched.copy()
    preds["game_id"] = _sanitize_numeric(preds.get("game_id"))
    if not markets.empty:
        join_cols_game = [c for c in ["game_id"] if c in preds.columns and c in markets.columns]
        merged = preds.merge(markets, on=join_cols_game, how="left", suffixes=("", "_m")) if join_cols_game else preds.copy()
        col = merged.get("market_spread_book")
        if col is None or col.isna().all():
            join_cols_matchup = [
                c for c in ["week", "home_team", "away_team"] if c in preds.columns and c in markets.columns
            ]
            if join_cols_matchup:
                merged = preds.merge(
                    markets,
                    on=join_cols_matchup,
                    how="left",
                    suffixes=("", "_m"),
                )
        preds = merged
    logger.debug(
        "build_predictions_for_year: post-market-merge rows=%s market_non_null=%s",
        len(preds),
        int(pd.to_numeric(preds.get("market_spread_book"), errors="coerce").notna().sum())
        if "market_spread_book" in preds.columns
        else 0,
    )

    if "neutral_site" in preds.columns:
        preds["neutral_site"] = preds["neutral_site"].fillna(0).astype(int)
    else:
        preds["neutral_site"] = 0
    preds["home_points"] = _sanitize_numeric(preds.get("home_points"))
    preds["away_points"] = _sanitize_numeric(preds.get("away_points"))

    home_rating = preds["home_team"].map(team_ratings).fillna(0.5)
    away_rating = preds["away_team"].map(team_ratings).fillna(0.5)

    # Convert ratings into point-spread with modest home-field bump
    base_diff = (home_rating - away_rating) * 15.0
    home_field = np.where(preds["neutral_site"] == 1, 0.0, 1.5)
    preds["model_spread_book"] = (base_diff + home_field).round(2)

    if "market_spread_book" in preds.columns:
        market_series = _sanitize_numeric(preds["market_spread_book"])
    else:
        market_series = pd.Series(np.nan, index=preds.index, dtype="float64")
    if "market_spread_fanduel" in preds.columns:
        preds["market_spread_fanduel"] = _sanitize_numeric(preds["market_spread_fanduel"])
    else:
        preds["market_spread_fanduel"] = pd.Series(np.nan, index=preds.index, dtype="float64")
    if "market_spread_cfbd" in preds.columns:
        preds["market_spread_cfbd"] = _sanitize_numeric(preds["market_spread_cfbd"])
    else:
        preds["market_spread_cfbd"] = pd.Series(np.nan, index=preds.index, dtype="float64")
    preds["market_spread_book"] = market_series
    market_missing = market_series.isna()
    preds["market_is_synthetic"] = market_missing.astype(int)

    if "market_spread_source" in preds.columns:
        preds["market_spread_source"] = (
            preds["market_spread_source"].fillna("").astype(str).str.strip().str.lower()
        )
    else:
        preds["market_spread_source"] = "model"

    preds.loc[
        preds["market_spread_source"].isin(["", "nan", "none"]),
        "market_spread_source",
    ] = "model"
    market_present = preds["market_spread_book"].notna()
    preds.loc[~market_present, "market_spread_source"] = "model"
    preds.loc[
        market_present & ~preds["market_spread_source"].isin(["fanduel", "cfbd"]),
        "market_spread_source",
    ] = "unknown"
    # Authoritative check: any row with a FanDuel numeric value should always tag as FanDuel.
    if "market_spread_fanduel" in preds.columns:
        fanduel_available = preds["market_spread_fanduel"].notna()
        preds.loc[fanduel_available, "market_spread_source"] = "fanduel"
    tol = 1e-6
    fd_mask = (
        market_present
        & preds["market_spread_fanduel"].notna()
        & (preds["market_spread_book"] - preds["market_spread_fanduel"]).abs() <= tol
    )
    preds.loc[fd_mask, "market_spread_source"] = "fanduel"
    cf_mask = (
        market_present
        & (preds["market_spread_source"] == "unknown")
        & preds["market_spread_cfbd"].notna()
        & (preds["market_spread_book"] - preds["market_spread_cfbd"]).abs() <= tol
    )
    preds.loc[cf_mask, "market_spread_source"] = "cfbd"

    # Use model spread as fall-back only for calculations so we avoid NaNs downstream
    market_for_calc = market_series.where(~market_missing, preds["model_spread_book"])
    preds["market_spread_effective"] = market_for_calc.round(2)

    preds["expected_market_spread_book"] = (
        market_for_calc * 0.6 + preds["model_spread_book"] * 0.4
    ).round(2)

    preds["edge_points_book"] = (
        preds["model_spread_book"] - market_for_calc
    ).round(2)
    preds["value_points_book"] = (
        market_for_calc - preds["expected_market_spread_book"]
    ).round(2)

    qualified_mask = (
        (preds["edge_points_book"].abs() >= 2.0)
        & (preds["value_points_book"].abs() >= 1.0)
        & (
            np.sign(preds["edge_points_book"]) == np.sign(preds["value_points_book"] * -1)
        )
    )
    preds["qualified_edge_flag"] = qualified_mask.astype(int)

    preds["model_pick_side"] = np.where(preds["edge_points_book"] > 0, "AWAY", "HOME")
    preds.loc[preds["edge_points_book"].abs() < 1e-6, "model_pick_side"] = ""

    preds["played"] = (
        preds["home_points"].notna() & preds["away_points"].notna()
    ).astype(int)

    # Order + select columns expected by downstream tables
    cols = [
        "week",
        "date",
        "kickoff_utc" if "kickoff_utc" in preds.columns else None,
        "away_team",
        "home_team",
        "neutral_site",
        "model_spread_book",
        "market_spread_book",
        "market_spread_fanduel",
        "market_spread_cfbd",
        "market_spread_source",
        "market_spread_effective",
        "expected_market_spread_book",
        "edge_points_book",
        "value_points_book",
        "qualified_edge_flag",
        "model_pick_side",
        "market_is_synthetic",
        "home_points",
        "away_points",
        "played",
        "game_id",
    ]
    cols = [c for c in cols if c in preds.columns]

    out = preds[cols].copy()
    out.sort_values(["week", "date", "home_team"], inplace=True, ignore_index=True)
    if "market_spread_fanduel" in out.columns and "market_spread_source" in out.columns:
        fd_available = pd.to_numeric(out["market_spread_fanduel"], errors="coerce").notna()
        if fd_available.any():
            out.loc[fd_available, "market_spread_source"] = "fanduel"

    synthetic_count = (
        int(out.get("market_is_synthetic", pd.Series(dtype=int)).sum())
        if "market_is_synthetic" in out.columns
        else 0
    )
    edge_series = pd.to_numeric(out.get("edge_points_book"), errors="coerce") if "edge_points_book" in out.columns else pd.Series(dtype="float64")
    non_zero_edges = int(edge_series.fillna(0).abs().gt(1e-9).sum())
    logger.debug(
        "build_predictions_for_year:end synthetic_rows=%s edge_nonzero=%s",
        synthetic_count,
        non_zero_edges,
    )
    debug_market = os.environ.get("DEBUG_MARKET", "0").strip().lower() in {"1", "true", "yes"}
    if debug_market:
        log_frame = preds.copy()
        for _, row in log_frame.iterrows():
            def _scalar(val: Any) -> Any:
                if isinstance(val, pd.Series):
                    for item in val:
                        if pd.notna(item):
                            return item
                    return val.iloc[0] if not val.empty else None
                return val

            fan_duel_val = _scalar(row.get("market_spread_fanduel"))
            fan_duel_raw = _scalar(row.get("market_spread_fanduel_raw"))
            cfbd_val = _scalar(row.get("market_spread_cfbd"))
            cfbd_raw = _scalar(row.get("market_spread_cfbd_raw"))
            book_val = _scalar(row.get("market_spread_book"))
            book_raw = _scalar(row.get("market_spread_book_raw"))

            extra_parts = []
            if pd.isna(fan_duel_val) and pd.notna(fan_duel_raw) and str(fan_duel_raw) != "":
                extra_parts.append(f"fan_duel_raw={fan_duel_raw!r}")
            if pd.isna(cfbd_val) and pd.notna(cfbd_raw) and str(cfbd_raw) != "":
                extra_parts.append(f"cfbd_raw={cfbd_raw!r}")
            if pd.isna(fan_duel_val) and pd.notna(book_raw) and str(book_raw) != "":
                extra_parts.append(f"book_raw={book_raw!r}")
            detail_suffix = (" " + " ".join(extra_parts)) if extra_parts else ""

            logger.debug(
                "market selection: week=%s game_id=%s home=%s away=%s fan_duel=%s cfbd=%s market=%s effective=%s source=%s synthetic=%s%s",
                row.get("week"),
                row.get("game_id"),
                row.get("home_team"),
                row.get("away_team"),
                fan_duel_val,
                cfbd_val,
                book_val,
                row.get("market_spread_effective"),
                row.get("market_spread_source"),
                row.get("market_is_synthetic"),
                detail_suffix,
            )

    return out


__all__ = ["build_predictions_for_year"]
