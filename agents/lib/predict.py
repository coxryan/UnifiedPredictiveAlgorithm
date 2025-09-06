from __future__ import annotations
import numpy as np
import pandas as pd

# Minimal model: uses precomputed team inputs, builds a model spread, and
# computes expected_market_spread_book via a simple calibration.
# IMPORTANT: We treat synthetic market (non-current weeks) as "no real market"
# for betting/qualification decisions.

def _calibrate_expected(model_spread: pd.Series) -> pd.Series:
    # Simple identity calibration placeholder; replace with your fitted mapping if present
    return model_spread.astype(float)

def build_predictions_and_edge(
    inputs_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    df = schedule_df.copy()

    # Example model spread: use placeholder if your real function lives elsewhere.
    # If you already have a better spread in schedule_df, keep it; else derive from inputs.
    if "model_spread_book" not in df.columns:
        # crude: neutral = 0, home edge small bump; replace with your real procedure
        df["model_spread_book"] = 0.0
        if "neutral_site" in df.columns:
            df.loc[df["neutral_site"].fillna(0).astype(int) == 0, "model_spread_book"] += 0.5

    # Expected market from model via calibration
    df["expected_market_spread_book"] = _calibrate_expected(df["model_spread_book"])

    # Diagnostics / NaN reasoning
    df["has_model"] = df["model_spread_book"].notna()
    # Treat synthetic as NOT having a real market for qualification
    synthetic = df.get("market_is_synthetic", pd.Series(False, index=df.index)).fillna(False)
    df["has_market"] = df["market_spread_book"].notna() & (~synthetic)

    # Edge/value (only meaningful when real market exists)
    df["edge_points_book"] = np.where(
        df["has_market"],
        (df["model_spread_book"] - df["market_spread_book"]).astype(float),
        np.nan,
    )
    df["value_points_book"] = np.where(
        df["has_market"],
        (df["expected_market_spread_book"] - df["market_spread_book"]).astype(float),
        np.nan,
    )

    # Simple qualifier: real market present AND |value| >= threshold
    VALUE_THRESH = 3.0
    df["qualified_edge_flag"] = df["has_market"] & (df["value_points_book"].abs() >= VALUE_THRESH)

    # nan_reason for debugging
    df["nan_reason"] = np.where(~df["has_market"], "no_real_market",
                         np.where(~df["has_model"], "no_model_value",
                         np.where(df["expected_market_spread_book"].isna(), "expected_calc_na",
                         np.where(df["edge_points_book"].isna(), "edge_calc_na",
                         np.where(df["value_points_book"].isna(), "value_calc_na", "")))))

    # Keep useful columns plus pass-throughs
    keep = [
        "game_id","week","date","away_team","home_team","neutral_site",
        "model_spread_book","market_spread_book","expected_market_spread_book",
        "edge_points_book","value_points_book","qualified_edge_flag",
        "market_is_synthetic","nan_reason"
    ]
    passthrough = [c for c in df.columns if c not in keep]
    return df[keep + passthrough]