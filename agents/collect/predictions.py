from __future__ import annotations

from typing import Optional, List, Dict

import logging
import os
import numpy as np
import pandas as pd

from .cache import ApiCache
from .schedule import discover_current_week
from .team_inputs import build_team_inputs_datadriven
from .spread_model import load_residual_model
from .model_dataset import FEATURE_COLUMNS, prepare_feature_frame
from .markets import get_market_lines_for_current_week
from .cfbd_clients import CfbdClients

logger = logging.getLogger(__name__)


_SOURCE_ADJ_SCALE = {
    "fanduel": 1.0,
    "cfbd": 0.35,
    "model": 0.0,
    "unknown": 0.5,
}

try:
    _MAX_MARKET_ADJUSTMENT = float(os.environ.get("MAX_MARKET_ADJUSTMENT", "8.0"))
except ValueError:
    _MAX_MARKET_ADJUSTMENT = 8.0


def _get_float_env(key: str, default: float | None = None) -> float | None:
    raw = os.environ.get(key)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _clamp(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


_EXPECTED_MARKET_LAMBDA = _get_float_env("EXPECTED_MARKET_LAMBDA", 0.6) or 0.6
_EXPECTED_MARKET_LAMBDA = _clamp(_EXPECTED_MARKET_LAMBDA, lo=0.0, hi=1.0)
_EDGE_POINTS_MIN_DEFAULT = 2.5
_EDGE_POINTS_MIN = _get_float_env("EDGE_POINTS_QUALIFY_MIN", _EDGE_POINTS_MIN_DEFAULT) or _EDGE_POINTS_MIN_DEFAULT
_EDGE_POINTS_MIN = max(0.0, _EDGE_POINTS_MIN)
_value_override = _get_float_env("VALUE_POINTS_QUALIFY_MIN")
if _value_override is None:
    _VALUE_POINTS_MIN = round(_EDGE_POINTS_MIN * max(0.0, 1.0 - _EXPECTED_MARKET_LAMBDA), 2)
else:
    _VALUE_POINTS_MIN = max(0.0, _value_override)
_CONFIDENCE_MIN_DEFAULT = 0.65
_CONFIDENCE_MIN = _get_float_env("CONFIDENCE_QUALIFY_MIN", _CONFIDENCE_MIN_DEFAULT) or _CONFIDENCE_MIN_DEFAULT
_CONFIDENCE_MIN = max(0.0, min(1.0, _CONFIDENCE_MIN))


def _sanitize_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _rating_from_team_inputs(team_inputs: pd.DataFrame) -> pd.Series:
    if team_inputs is None or team_inputs.empty:
        return pd.Series(dtype="float64")
    df = team_inputs.copy()
    base_cols = [
        "wrps_percent_0_100",
        "talent_score_0_100",
        "srs_score_0_100",
        "stat_off_index_0_100",
        "stat_def_index_0_100",
        "stat_st_index_0_100",
    ]
    for col in base_cols:
        if col in df.columns:
            df[col] = _sanitize_numeric(df[col])
        else:
            df[col] = 0.0
    if "stat_off_index_0_100" not in df.columns or df["stat_off_index_0_100"].isna().all():
        off_cols = [c for c in [
            "stat_off_ppg",
            "stat_off_ypp",
            "stat_off_success",
            "stat_off_explosiveness",
        ] if c in df.columns]
        if off_cols:
            df["stat_off_index_0_100"] = df[off_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    if "stat_def_index_0_100" not in df.columns or df["stat_def_index_0_100"].isna().all():
        def_cols = [c for c in [
            "stat_def_ppg",
            "stat_def_ypp",
            "stat_def_success",
            "stat_def_explosiveness",
        ] if c in df.columns]
        if def_cols:
            df["stat_def_index_0_100"] = df[def_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    if "stat_st_index_0_100" not in df.columns or df["stat_st_index_0_100"].isna().all():
        st_cols = [c for c in ["stat_st_points_per_play"] if c in df.columns]
        if st_cols:
            df["stat_st_index_0_100"] = df[st_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    # Weighted composite score (0-1 scale)
    w_wrps = df["wrps_percent_0_100"].fillna(50.0) / 100.0
    w_talent = df["talent_score_0_100"].fillna(50.0) / 100.0
    w_srs = df["srs_score_0_100"].fillna(50.0) / 100.0
    w_off = df["stat_off_index_0_100"].fillna(50.0) / 100.0
    w_def = df["stat_def_index_0_100"].fillna(50.0) / 100.0
    w_st = df["stat_st_index_0_100"].fillna(50.0) / 100.0

    rating = (
        0.30 * w_wrps
        + 0.30 * w_talent
        + 0.20 * w_srs
        + 0.10 * w_off
        + 0.07 * w_def
        + 0.03 * w_st
    )
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
    scoreboard_df: Optional[pd.DataFrame] = None,
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

    include_grade_features = os.environ.get("INCLUDE_GRADE_FEATURES", "0").strip().lower() in {"1", "true", "yes", "y"}
    grade_columns: List[str] = []
    grade_lookup: Dict[str, pd.Series] = {}
    if (
        include_grade_features
        and team_inputs_df is not None
        and not team_inputs_df.empty
        and "team" in team_inputs_df.columns
    ):
        grade_columns = [c for c in team_inputs_df.columns if c.startswith("grade_")]
        if grade_columns:
            indexed_inputs = team_inputs_df.set_index("team")
            for col in grade_columns:
                if col in indexed_inputs.columns:
                    grade_lookup[col] = indexed_inputs[col]

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

    if scoreboard_df is not None and not scoreboard_df.empty:
        sb = scoreboard_df.copy()
        for col in ("home_school", "away_school", "home_team", "away_team"):
            if col in sb.columns:
                sb[col] = sb[col].astype(str).str.strip()
        home_series = sb.get("home_school")
        home_team_series = sb.get("home_team")
        if home_series is None:
            home_series = home_team_series
        elif home_team_series is not None:
            home_series = home_series.combine_first(home_team_series)

        away_series = sb.get("away_school")
        away_team_series = sb.get("away_team")
        if away_series is None:
            away_series = away_team_series
        elif away_team_series is not None:
            away_series = away_series.combine_first(away_team_series)

        sb["home_norm"] = (
            home_series.astype(str).str.strip().str.casefold()
            if home_series is not None
            else pd.Series("", index=sb.index)
        )
        sb["away_norm"] = (
            away_series.astype(str).str.strip().str.casefold()
            if away_series is not None
            else pd.Series("", index=sb.index)
        )
        sb["date_key"] = sb.get("date").astype(str).str.slice(0, 10)
        sb["home_points"] = pd.to_numeric(sb.get("home_points"), errors="coerce")
        sb["away_points"] = pd.to_numeric(sb.get("away_points"), errors="coerce")
        sb = sb.dropna(subset=["home_norm", "away_norm"])

        preds["home_norm"] = preds["home_team"].astype(str).str.strip().str.casefold()
        preds["away_norm"] = preds["away_team"].astype(str).str.strip().str.casefold()
        preds["date_key"] = preds.get("kickoff_utc", preds.get("date", "")).astype(str).str.slice(0, 10)
        preds.loc[
            preds["date_key"].isna() | (preds["date_key"] == ""), "date_key"
        ] = preds.get("date").astype(str).str.slice(0, 10)

        sb_latest = (
            sb.sort_values("date_key")
            [["date_key", "home_norm", "away_norm", "home_points", "away_points"]]
            .drop_duplicates(subset=["date_key", "home_norm", "away_norm"], keep="last")
        )
        preds = preds.merge(
            sb_latest,
            on=["date_key", "home_norm", "away_norm"],
            how="left",
            suffixes=("", "_sb"),
        )
        if "home_points_sb" in preds.columns:
            preds["home_points"] = preds.get("home_points").where(
                preds.get("home_points").notna(), preds.get("home_points_sb")
            )
            preds["away_points"] = preds.get("away_points").where(
                preds.get("away_points").notna(), preds.get("away_points_sb")
            )
            preds = preds.drop(columns=["home_points_sb", "away_points_sb"], errors="ignore")

        missing_mask = preds["home_points"].isna() | preds["away_points"].isna()
        if missing_mask.any():
            sb_teams = (
                sb.sort_values("date_key")
                [["home_norm", "away_norm", "home_points", "away_points"]]
                .drop_duplicates(subset=["home_norm", "away_norm"], keep="last")
            )
            preds = preds.merge(
                sb_teams,
                on=["home_norm", "away_norm"],
                how="left",
                suffixes=("", "_sb2"),
            )
            preds.loc[preds["home_points"].isna(), "home_points"] = preds.loc[
                preds["home_points"].isna(), "home_points_sb2"
            ]
            preds.loc[preds["away_points"].isna(), "away_points"] = preds.loc[
                preds["away_points"].isna(), "away_points_sb2"
            ]
            preds = preds.drop(columns=["home_points_sb2", "away_points_sb2"], errors="ignore")

        preds = preds.drop(columns=["home_norm", "away_norm", "date_key"], errors="ignore")

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

    if grade_lookup:
        for col, series in grade_lookup.items():
            home_col = f"home_{col}"
            away_col = f"away_{col}"
            try:
                dedup = series[~series.index.duplicated(keep="last")]
                home_values = preds["home_team"].map(dedup)
                away_values = preds["away_team"].map(dedup)
                preds[home_col] = home_values
                preds[away_col] = away_values
            except Exception:
                preds[home_col] = preds["home_team"].map(series)
                preds[away_col] = preds["away_team"].map(series)

    home_rating = preds["home_team"].map(team_ratings).fillna(0.5)
    away_rating = preds["away_team"].map(team_ratings).fillna(0.5)

    base_diff = (home_rating - away_rating) * 15.0
    home_field = np.where(preds["neutral_site"] == 1, 0.0, 1.5)
    baseline_model = base_diff + home_field

    preds["model_spread_baseline"] = baseline_model.round(2)
    preds["model_spread_book"] = preds["model_spread_baseline"]

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

    preds["_row_id"] = np.arange(len(preds))
    residual_model = load_residual_model()
    if residual_model is not None and not preds.empty:
        feature_ready, _ = prepare_feature_frame(preds, team_inputs=team_inputs_df)
        preds = preds.drop(
            columns=[
                "market_adjustment",
                "market_adjustment_raw",
                "market_adjustment_linear",
                "market_adjustment_nonlinear",
                "residual_pred_raw",
                "residual_pred_calibrated",
                "residual_pred_linear",
                "residual_pred_nonlinear",
            ],
            errors="ignore",
        )
        has_features = bool(residual_model.features)
        if has_features and not feature_ready.empty:
            normalized_features = residual_model.transform_features(feature_ready)
            X = normalized_features[residual_model.features].to_numpy(dtype=float)
            components = residual_model.predict_components(X)
            residual_df = pd.DataFrame(
                {
                    "_row_id": feature_ready["_row_id"],
                    "residual_pred_linear": components["ridge"],
                    "residual_pred_nonlinear": components["gbdt"],
                    "residual_pred_raw": components["raw"],
                    "residual_pred_calibrated": components["calibrated"],
                }
            )
            residual_df = residual_df.drop_duplicates(subset=["_row_id"])
            preds = preds.merge(residual_df, on="_row_id", how="left")
            raw_linear = preds["residual_pred_linear"].fillna(0.0)
            raw_nonlinear = preds["residual_pred_nonlinear"].fillna(0.0)
            raw_uncalibrated = preds["residual_pred_raw"].fillna(0.0)
            raw_calibrated = preds["residual_pred_calibrated"].fillna(0.0)

            preds[[
                "residual_pred_linear",
                "residual_pred_nonlinear",
                "residual_pred_raw",
                "residual_pred_calibrated",
            ]] = preds[[
                "residual_pred_linear",
                "residual_pred_nonlinear",
                "residual_pred_raw",
                "residual_pred_calibrated",
            ]].fillna(0.0).round(3)

            preds["market_adjustment_linear"] = (-raw_linear).round(2)
            preds["market_adjustment_nonlinear"] = (-raw_nonlinear).round(2)
            preds["market_adjustment_raw"] = (-raw_uncalibrated).round(2)

            sigma = max(residual_model.residual_std, 1e-6)
            confidence = np.exp(-raw_calibrated.abs() / (sigma * 1.5))
            confidence = confidence.where(
                ~preds["market_is_synthetic"].astype(bool),
                confidence * 0.5,
            )
            confidence = confidence.where(
                preds["market_spread_source"].isin(["fanduel", "cfbd"]),
                confidence * 0.7,
            )
            confidence = confidence.clip(0.0, 0.99)

            source_scale = preds["market_spread_source"].map(_SOURCE_ADJ_SCALE).fillna(0.5)
            scaled_delta = source_scale * confidence
            market_adjustment = (-raw_calibrated * scaled_delta).clip(
                -_MAX_MARKET_ADJUSTMENT,
                _MAX_MARKET_ADJUSTMENT,
            )
            preds["market_adjustment"] = market_adjustment.round(2)
            preds["model_confidence"] = confidence.round(3)
            preds["model_residual_sigma"] = residual_model.residual_std

            market_present_mask = preds["market_spread_book"].notna()
            preds.loc[market_present_mask, "model_spread_book"] = (
                preds.loc[market_present_mask, "market_spread_book"]
                + preds.loc[market_present_mask, "market_adjustment"]
            ).round(2)

            missing_market_mask = ~market_present_mask
            if missing_market_mask.any():
                preds.loc[missing_market_mask, "market_adjustment"] = 0.0
                preds.loc[missing_market_mask, "model_confidence"] = 0.0
        else:
            preds["market_adjustment"] = 0.0
            preds["market_adjustment_raw"] = 0.0
            preds["market_adjustment_linear"] = 0.0
            preds["market_adjustment_nonlinear"] = 0.0
            preds["residual_pred_linear"] = 0.0
            preds["residual_pred_nonlinear"] = 0.0
            preds["residual_pred_raw"] = 0.0
            preds["residual_pred_calibrated"] = 0.0
            preds["model_confidence"] = 0.0
            preds["model_residual_sigma"] = residual_model.residual_std
    else:
        preds["market_adjustment"] = 0.0
        preds["market_adjustment_raw"] = 0.0
        preds["market_adjustment_linear"] = 0.0
        preds["market_adjustment_nonlinear"] = 0.0
        preds["residual_pred_linear"] = 0.0
        preds["residual_pred_nonlinear"] = 0.0
        preds["residual_pred_raw"] = 0.0
        preds["residual_pred_calibrated"] = 0.0
        preds["model_confidence"] = 0.0
        preds["model_residual_sigma"] = np.nan

    # Use model spread as fall-back only for calculations so we avoid NaNs downstream
    market_for_calc = market_series.where(~market_missing, preds["model_spread_book"])
    preds["market_spread_effective"] = market_for_calc.round(2)

    preds["expected_market_spread_book"] = (
        market_for_calc * _EXPECTED_MARKET_LAMBDA
        + preds["model_spread_book"] * (1.0 - _EXPECTED_MARKET_LAMBDA)
    ).round(2)

    preds["edge_points_book"] = (
        preds["model_spread_book"] - market_for_calc
    ).round(2)
    preds["value_points_book"] = (
        market_for_calc - preds["expected_market_spread_book"]
    ).round(2)

    qualified_mask = (
        (preds["edge_points_book"].abs() >= _EDGE_POINTS_MIN)
        & (preds["value_points_book"].abs() >= _VALUE_POINTS_MIN)
        & (preds["model_confidence"] >= _CONFIDENCE_MIN)
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

    if "_row_id" in preds.columns:
        preds = preds.drop(columns=["_row_id"])

    # Order + select columns expected by downstream tables
    cols = [
        "week",
        "date",
        "kickoff_utc" if "kickoff_utc" in preds.columns else None,
        "away_team",
        "home_team",
        "neutral_site",
        "model_spread_book",
        "model_spread_baseline",
        "market_adjustment",
        "market_adjustment_raw",
        "market_adjustment_linear",
        "market_adjustment_nonlinear",
        "residual_pred_calibrated",
        "residual_pred_raw",
        "residual_pred_linear",
        "residual_pred_nonlinear",
        "model_confidence",
        "model_residual_sigma",
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
        "home_grade_qb_letter",
        "away_grade_qb_letter",
        "home_grade_qb_score",
        "away_grade_qb_score",
        "home_grade_qb_percentile",
        "away_grade_qb_percentile",
        "home_grade_wr_letter",
        "away_grade_wr_letter",
        "home_grade_wr_score",
        "away_grade_wr_score",
        "home_grade_wr_percentile",
        "away_grade_wr_percentile",
        "home_grade_rb_letter",
        "away_grade_rb_letter",
        "home_grade_rb_score",
        "away_grade_rb_score",
        "home_grade_rb_percentile",
        "away_grade_rb_percentile",
        "home_grade_ol_letter",
        "away_grade_ol_letter",
        "home_grade_ol_score",
        "away_grade_ol_score",
        "home_grade_ol_percentile",
        "away_grade_ol_percentile",
        "home_grade_dl_letter",
        "away_grade_dl_letter",
        "home_grade_dl_score",
        "away_grade_dl_score",
        "home_grade_dl_percentile",
        "away_grade_dl_percentile",
        "home_grade_lb_letter",
        "away_grade_lb_letter",
        "home_grade_lb_score",
        "away_grade_lb_score",
        "home_grade_lb_percentile",
        "away_grade_lb_percentile",
        "home_grade_db_letter",
        "away_grade_db_letter",
        "home_grade_db_score",
        "away_grade_db_score",
        "home_grade_db_percentile",
        "away_grade_db_percentile",
        "home_grade_st_letter",
        "away_grade_st_letter",
        "home_grade_st_score",
        "away_grade_st_score",
        "home_grade_st_percentile",
        "away_grade_st_percentile",
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
