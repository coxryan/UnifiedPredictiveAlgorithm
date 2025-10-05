from __future__ import annotations

from typing import Optional

import pandas as pd

from agents.storage import read_dataset


def build_live_edge_report(
    year: int,
    preds_df: Optional[pd.DataFrame] = None,
    preds_csv: Optional[str] = None,
    max_rows: int = 100,
) -> pd.DataFrame:
    """Return a trimmed table of future games with largest edge/value combos."""

    if preds_df is None:
        if not preds_csv:
            return pd.DataFrame(
                columns=[
                    "week",
                    "date",
                    "away_team",
                    "home_team",
                    "edge_points_book",
                    "value_points_book",
                    "qualified_edge_flag",
                ]
            )
        try:
            preds_df = read_dataset("upa_predictions")
        except Exception:
            return pd.DataFrame(
                columns=[
                    "week",
                    "date",
                    "away_team",
                    "home_team",
                    "edge_points_book",
                    "value_points_book",
                    "qualified_edge_flag",
                ]
            )

    if preds_df is None or preds_df.empty:
        return pd.DataFrame(
            columns=[
                "week",
                "date",
                "away_team",
                "home_team",
                "edge_points_book",
                "value_points_book",
                "qualified_edge_flag",
            ]
        )

    df = preds_df.copy()
    for col in ["edge_points_book", "value_points_book", "market_spread_book", "model_spread_book"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["qualified_edge_flag"] = df.get("qualified_edge_flag", 0).astype(int)

    played = df.get("played")
    if played is not None:
        mask_future = ~(pd.to_numeric(played, errors="coerce").fillna(0).astype(int).astype(bool))
    else:
        mask_future = ~(
            pd.to_numeric(df.get("home_points"), errors="coerce").notna()
            & pd.to_numeric(df.get("away_points"), errors="coerce").notna()
        )
    future = df.loc[mask_future].copy()

    if future.empty:
        return future[[c for c in ["week", "date", "away_team", "home_team", "edge_points_book", "value_points_book", "qualified_edge_flag"] if c in future.columns]]

    future["abs_edge"] = future["edge_points_book"].abs()
    future["abs_value"] = future["value_points_book"].abs()

    future.sort_values(
        by=["qualified_edge_flag", "abs_edge", "abs_value", "date"],
        ascending=[False, False, False, True],
        inplace=True,
        ignore_index=True,
    )

    cols = [
        "week",
        "date",
        "away_team",
        "home_team",
        "model_spread_book" if "model_spread_book" in future.columns else None,
        "market_spread_book" if "market_spread_book" in future.columns else None,
        "edge_points_book",
        "value_points_book",
        "qualified_edge_flag",
        "model_pick_side" if "model_pick_side" in future.columns else None,
    ]
    cols = [c for c in cols if c in future.columns]

    out = future[cols].head(max_rows).copy()
    out.reset_index(drop=True, inplace=True)
    return out


__all__ = ["build_live_edge_report"]
