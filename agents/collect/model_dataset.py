from __future__ import annotations

from typing import Iterable

import pandas as pd

from agents.storage import read_dataset


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


def load_training_dataset() -> pd.DataFrame:
    preds = read_dataset("upa_predictions")
    if preds.empty:
        return preds
    preds = preds.copy()
    preds["home_cover"] = pd.to_numeric(preds.get("home_points"), errors="coerce")
    preds["away_cover"] = pd.to_numeric(preds.get("away_points"), errors="coerce")
    preds["market_spread_book"] = pd.to_numeric(preds.get("market_spread_book"), errors="coerce")
    preds["model_spread_book"] = pd.to_numeric(preds.get("model_spread_book"), errors="coerce")
    preds["edge_points_book"] = pd.to_numeric(preds.get("edge_points_book"), errors="coerce")

    team_inputs = read_dataset("upa_team_inputs_datadriven_v0")
    if team_inputs.empty:
        return pd.DataFrame()
    team_inputs = team_inputs.copy()
    team_inputs["team"] = team_inputs["team"].astype(str)
    cols = ["team"] + [c for c in FEATURE_COLUMNS if c in team_inputs.columns]
    if "season" in team_inputs.columns:
        cols.insert(1, "season")
    team_inputs = team_inputs[cols]

    def _merge(side: str) -> pd.DataFrame:
        suffix = "_home" if side == "home" else "_away"
        ignore_cols = {"team", "season"} if "season" in team_inputs.columns else {"team"}
        subset = team_inputs.rename(columns={c: f"{c}{suffix}" for c in team_inputs.columns if c not in ignore_cols})
        subset = subset.rename(columns={"team": f"{side}_team"})
        return subset

    merged = preds.merge(_merge("home"), on="home_team", how="left")
    merged = merged.merge(_merge("away"), on="away_team", how="left")
    return merged


__all__ = ["load_training_dataset", "FEATURE_COLUMNS"]
