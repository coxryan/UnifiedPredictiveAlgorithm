from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from agents.storage import read_dataset

# Base feature columns that exist in the team inputs table. These remain the
# foundation for delta-style features when we assemble the residual dataset.
FEATURE_COLUMNS: Iterable[str] = (
    "wrps_percent_0_100",
    "talent_score_0_100",
    "srs_score_0_100",
    "stat_off_index_0_100",
    "stat_def_index_0_100",
    "stat_st_index_0_100",
    "grade_qb_percentile",
    "grade_rb_percentile",
    "grade_wr_percentile",
    "grade_ol_percentile",
    "grade_dl_percentile",
    "grade_lb_percentile",
    "grade_db_percentile",
    "grade_st_percentile",
)

# Additional engineered features backed by historical results. We derive these on
# a per-team rolling basis and then take the home-away delta when training.
ROLLING_FEATURES: Sequence[str] = (
    "rolling_margin_3",
    "rolling_margin_5",
    "rolling_residual_3",
    "rolling_residual_5",
    "rolling_market_3",
    "rolling_market_5",
    "days_since_game",
)


@dataclass(frozen=True)
class TrainingDataset:
    """Container for the residual training frame and metadata used downstream."""

    frame: pd.DataFrame
    # Columns used as model inputs after preprocessing.
    feature_columns: Sequence[str]
    # Target column containing the residual delta the model is expected to learn.
    target_column: str = "residual_target"


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_datetime_columns(df: pd.DataFrame) -> pd.Series:
    """Return a unified kickoff timestamp for ordering games chronologically."""

    kickoff_columns = [
        "kickoff_utc",
        "start_date",
        "datetime",
        "kickoff_time",
        "date",
    ]
    kickoff = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in kickoff_columns:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        kickoff = kickoff.fillna(parsed)
    # Some historical rows only have a `date` column without timezone. Ensure we
    # at least parse that string even if the previous loop already attempted it.
    if kickoff.isna().any() and "date" in df.columns:
        parsed_date = pd.to_datetime(df["date"], errors="coerce", utc=True)
        kickoff = kickoff.fillna(parsed_date)
    return kickoff


def _season_from_timestamp(ts: pd.Timestamp) -> int | float:
    if pd.isna(ts):
        return math.nan
    year = ts.year
    # College football seasons roll over in January bowls. Treat games before
    # July as part of the previous season.
    if ts.month < 7:
        return year - 1
    return year


def _season_series(preds: pd.DataFrame, kickoff: pd.Series) -> pd.Series:
    if "season" in preds.columns:
        season = _coerce_numeric(preds["season"]).round().astype("Int64")
        if season.notna().any():
            season_filled = season.astype(float)
            missing_mask = season_filled.isna()
            season_filled[missing_mask] = kickoff[missing_mask].map(_season_from_timestamp)
            return season_filled
    return kickoff.map(_season_from_timestamp)


def _build_team_inputs_frame(
    feature_cols: Sequence[str],
    team_inputs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source = team_inputs.copy() if team_inputs is not None else read_dataset("upa_team_inputs_datadriven_v0")
    if source.empty:
        return source

    df = source.copy()
    df["team"] = df["team"].astype(str)
    if "season" in df.columns:
        df["season"] = _coerce_numeric(df["season"]).astype("Int64")
    for col in feature_cols:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])
    keep_cols: List[str] = ["team"]
    if "season" in df.columns:
        keep_cols.append("season")
    keep_cols.extend([c for c in feature_cols if c in df.columns])
    return df[keep_cols]


def _merge_team_features(
    preds: pd.DataFrame,
    team_inputs: pd.DataFrame,
    side: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    if team_inputs.empty:
        return preds

    suffix = "_home" if side == "home" else "_away"
    left_keys = [f"{side}_team"]
    right_keys = ["team"]
    if "season" in team_inputs.columns and "season" in preds.columns:
        left_keys.append("season")
        right_keys.append("season")

    rename_map = {c: f"{c}{suffix}" for c in feature_cols if c in team_inputs.columns}
    renamed = team_inputs.rename(columns=rename_map)

    merged = preds.merge(
        renamed,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
    )
    # Drop the helper key columns that only exist on the right side.
    for key in right_keys:
        if key not in left_keys and key in merged.columns:
        merged = merged.drop(columns=key)
    if "season_x" in merged.columns and "season_y" in merged.columns:
        merged["season"] = merged["season_x"].combine_first(merged["season_y"])
        merged = merged.drop(columns=["season_x", "season_y"])
    elif "season_x" in merged.columns:
        merged = merged.rename(columns={"season_x": "season"})
    elif "season_y" in merged.columns:
        merged = merged.rename(columns={"season_y": "season"})
    return merged


def _team_long_frame(preds: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "game_id",
        "season",
        "week",
        "kickoff_dt",
        "market_spread_book",
        "market_spread_source",
        "market_is_synthetic",
        "actual_margin",
        "residual_target",
    ]
    base_cols = [c for c in base_cols if c in preds.columns]
    rows = []
    for side in ("home", "away"):
        cols = base_cols + [f"{side}_team", f"{('away' if side=='home' else 'home')}_team"]
        cols = [c for c in cols if c in preds.columns]
        sub = preds[cols].copy()
        sub["team"] = preds[f"{side}_team"].astype(str)
        other_side = "away" if side == "home" else "home"
        if f"{other_side}_team" in preds.columns:
            sub["opponent"] = preds[f"{other_side}_team"].astype(str)
        sub["is_home"] = 1 if side == "home" else 0
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _attach_rolling_features(preds: pd.DataFrame) -> pd.DataFrame:
    long_df = _team_long_frame(preds)
    if long_df.empty:
        return preds

    long_df = long_df.sort_values(["team", "kickoff_dt", "game_id"], kind="mergesort")

    def rolling_mean(series: pd.Series, window: int) -> pd.Series:
        return series.shift(1).rolling(window=window, min_periods=1).mean()

    for window in (3, 5):
        long_df[f"rolling_margin_{window}"] = (
            long_df.groupby("team")["actual_margin"].transform(lambda s: rolling_mean(s, window))
        )
        long_df[f"rolling_residual_{window}"] = (
            long_df.groupby("team")["residual_target"].transform(lambda s: rolling_mean(s, window))
        )
        long_df[f"rolling_market_{window}"] = (
            long_df.groupby("team")["market_spread_book"].transform(lambda s: rolling_mean(s, window))
        )

    long_df["days_since_game"] = long_df.groupby("team")["kickoff_dt"].transform(
        lambda s: s.diff().dt.total_seconds() / 86400.0
    )

    feature_cols = [c for c in ROLLING_FEATURES if c in long_df.columns]
    merged = preds.copy()
    for side in ("home", "away"):
        suffix = "_home" if side == "home" else "_away"
        subset = long_df.loc[long_df["is_home"] == (1 if side == "home" else 0)]
        keep = ["team", "game_id"] + feature_cols
        keep = [c for c in keep if c in subset.columns]
        if not keep:
            continue
        renamed = subset[keep].rename(columns={c: f"{c}{suffix}" for c in feature_cols})
        renamed = renamed.rename(columns={"team": f"{side}_team"})
        feature_cols_with_suffix = [f"{c}{suffix}" for c in feature_cols]
        join_cols = [col for col in renamed.columns if col not in feature_cols_with_suffix]
        merged = merged.merge(renamed, on=join_cols, how="left")
    return merged


def _season_normalize(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    if not cols:
        return df
    out = df.copy()
    if "season" not in out.columns:
        return out
    for season, group_idx in out.groupby("season").groups.items():
        idx = list(group_idx)
        subset = out.loc[idx, cols]
        mean = subset.mean(skipna=True)
        std = subset.std(skipna=True)
        std.replace(0.0, np.nan, inplace=True)
        normalized = (subset - mean) / std
        out.loc[idx, cols] = normalized
    out[cols] = out[cols].fillna(0.0)
    return out


def _encode_market_source(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "market_spread_source" not in out.columns:
        return out
    source = out["market_spread_source"].fillna("").astype(str).str.lower()
    for key in ("fanduel", "cfbd", "model", "unknown"):
        out[f"market_source_{key}"] = (source == key).astype(float)
    return out


def prepare_feature_frame(
    preds: pd.DataFrame,
    *,
    team_inputs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, List[str]]:
    if preds is None or preds.empty:
        return pd.DataFrame(), []

    df = preds.copy()
    df["home_points"] = _coerce_numeric(df.get("home_points"))
    df["away_points"] = _coerce_numeric(df.get("away_points"))
    df["market_spread_book"] = _coerce_numeric(df.get("market_spread_book"))
    df["week"] = _coerce_numeric(df.get("week")).astype("Int64") if "week" in df.columns else np.nan

    df["kickoff_dt"] = _parse_datetime_columns(df)
    df["season"] = _season_series(df, df["kickoff_dt"])

    df["actual_margin"] = df["home_points"] - df["away_points"]
    df["residual_target"] = df["actual_margin"] + df["market_spread_book"]

    team_inputs_frame = _build_team_inputs_frame(FEATURE_COLUMNS, team_inputs)
    for side in ("home", "away"):
        df = _merge_team_features(df, team_inputs_frame, side, FEATURE_COLUMNS)

    df = _attach_rolling_features(df)
    df = _encode_market_source(df)

    delta_features: List[str] = []
    for col in FEATURE_COLUMNS:
        home_col = f"{col}_home"
        away_col = f"{col}_away"
        if home_col in df.columns:
            df[home_col] = _coerce_numeric(df[home_col])
        if away_col in df.columns:
            df[away_col] = _coerce_numeric(df[away_col])
        if home_col in df.columns and away_col in df.columns:
            delta_col = f"delta_{col}"
            df[delta_col] = df[home_col].fillna(0.0) - df[away_col].fillna(0.0)
            delta_features.append(delta_col)

    for roll_col in ROLLING_FEATURES:
        home_col = f"{roll_col}_home"
        away_col = f"{roll_col}_away"
        if home_col in df.columns:
            df[home_col] = _coerce_numeric(df[home_col])
        if away_col in df.columns:
            df[away_col] = _coerce_numeric(df[away_col])
        if home_col in df.columns and away_col in df.columns:
            delta_col = f"delta_{roll_col}"
            df[delta_col] = df[home_col].fillna(0.0) - df[away_col].fillna(0.0)
            delta_features.append(delta_col)

    base_numeric = [
        "market_spread_book",
        "week" if "week" in df.columns else None,
        "neutral_site" if "neutral_site" in df.columns else None,
        "market_is_synthetic" if "market_is_synthetic" in df.columns else None,
    ]
    base_numeric = [c for c in base_numeric if c and c in df.columns]
    for col in base_numeric:
        df[col] = _coerce_numeric(df[col])

    feature_columns: List[str] = []
    feature_columns.extend(delta_features)
    feature_columns.extend(base_numeric)
    feature_columns.extend(col for col in df.columns if col.startswith("market_source_"))

    df[feature_columns] = df[feature_columns].fillna(0.0)

    if "week" in df.columns:
        week_numeric = df["week"].fillna(0).astype(int).astype(str)
    else:
        week_numeric = pd.Series(["0"] * len(df), index=df.index)
    df["cv_key"] = (
        df["season"].fillna(0).astype(int).astype(str)
        + "_"
        + week_numeric
    )

    return df, feature_columns


def load_training_dataset() -> TrainingDataset:
    preds = read_dataset("upa_predictions")
    if preds.empty:
        return TrainingDataset(frame=pd.DataFrame(), feature_columns=[])

    prepared, feature_columns = prepare_feature_frame(preds)
    if prepared.empty:
        return TrainingDataset(frame=pd.DataFrame(), feature_columns=[])

    mask = prepared["actual_margin"].notna() & prepared["market_spread_book"].notna()
    if "home_points" in prepared.columns and "away_points" in prepared.columns:
        mask &= prepared["home_points"].notna() & prepared["away_points"].notna()

    df = prepared.loc[mask].reset_index(drop=True)
    if df.empty:
        return TrainingDataset(frame=pd.DataFrame(), feature_columns=[])

    training_columns = [
        "season",
        "week" if "week" in df.columns else None,
        "cv_key",
        "game_id" if "game_id" in df.columns else None,
        "home_team" if "home_team" in df.columns else None,
        "away_team" if "away_team" in df.columns else None,
        "market_spread_book",
        "actual_margin",
        "residual_target",
    ]
    training_columns = [c for c in training_columns if c and c in df.columns]
    ordered_cols = training_columns + [c for c in feature_columns if c not in training_columns]
    df = df[ordered_cols].copy()

    return TrainingDataset(frame=df, feature_columns=feature_columns)


__all__ = [
    "TrainingDataset",
    "load_training_dataset",
    "prepare_feature_frame",
    "FEATURE_COLUMNS",
    "ROLLING_FEATURES",
]
