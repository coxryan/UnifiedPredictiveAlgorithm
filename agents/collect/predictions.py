from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .cache import ApiCache
from .schedule import discover_current_week
from .team_inputs import build_team_inputs_datadriven
from .markets import get_market_lines_for_current_week
from .cfbd_clients import CfbdClients


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
    # keep consistent index by team name
    rating.index = df["team"].astype(str)
    return rating


def _market_lookup(markets: pd.DataFrame) -> pd.DataFrame:
    if markets is None or markets.empty:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "market_spread_book"])
    df = markets.copy()
    for col in ("game_id", "week"):
        if col in df.columns:
            df[col] = _sanitize_numeric(df[col])
    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df[[c for c in ["game_id", "week", "home_team", "away_team", "market_spread_book"] if c in df.columns]].drop_duplicates()


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

    if markets_df is None:
        current_week = discover_current_week(sched) or int(_sanitize_numeric(sched["week"]).max() or 1)
        markets_df = get_market_lines_for_current_week(year, int(current_week), sched, apis, cache)

    markets = _market_lookup(markets_df)

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

    preds["neutral_site"] = preds.get("neutral_site", 0).fillna(0).astype(int)
    preds["home_points"] = _sanitize_numeric(preds.get("home_points"))
    preds["away_points"] = _sanitize_numeric(preds.get("away_points"))

    home_rating = preds["home_team"].map(team_ratings).fillna(0.5)
    away_rating = preds["away_team"].map(team_ratings).fillna(0.5)

    # Convert ratings into point-spread with modest home-field bump
    base_diff = (home_rating - away_rating) * 15.0
    home_field = np.where(preds["neutral_site"] == 1, 0.0, 1.5)
    preds["model_spread_book"] = (base_diff + home_field).round(2)

    preds["market_spread_book"] = _sanitize_numeric(preds.get("market_spread_book"))
    preds["market_is_synthetic"] = preds["market_spread_book"].isna()
    preds.loc[preds["market_is_synthetic"], "market_spread_book"] = preds.loc[
        preds["market_is_synthetic"], "model_spread_book"
    ].round(2)

    # Expected market = smoothed blend of model + historical market (here just a dampened model)
    preds["expected_market_spread_book"] = (
        preds["market_spread_book"] * 0.6 + preds["model_spread_book"] * 0.4
    ).round(2)

    preds["edge_points_book"] = (preds["model_spread_book"] - preds["market_spread_book"]).round(2)
    preds["value_points_book"] = (
        preds["market_spread_book"] - preds["expected_market_spread_book"]
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
    return out


__all__ = ["build_predictions_for_year"]
