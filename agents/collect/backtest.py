from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .cache import ApiCache
from .cfbd_clients import CfbdClients
from .schedule import load_schedule_for_year
from .team_inputs import build_team_inputs_datadriven
from .markets import get_market_lines_for_current_week
from .predictions import build_predictions_for_year
from .helpers import _apply_book_grades, _mirror_book_to_legacy_columns
from agents.storage import read_dataset, write_dataset

logger = logging.getLogger(__name__)


def _compute_backtest_summary(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "games",
                "wins",
                "losses",
                "pushes",
                "hit_rate",
                "qualified_games",
                "qualified_wins",
                "qualified_losses",
                "qualified_pushes",
                "qualified_hit_rate",
            ]
        )

    data = df.copy()
    data["week"] = pd.to_numeric(data.get("week"), errors="coerce")

    played_series = data["played"] if "played" in data.columns else pd.Series(0, index=data.index)
    data["played"] = pd.to_numeric(played_series, errors="coerce").fillna(0).astype(int)

    qual_series = (
        data["qualified_edge_flag"]
        if "qualified_edge_flag" in data.columns
        else pd.Series(0, index=data.index)
    )
    data["qualified_edge_flag"] = (
        pd.to_numeric(qual_series, errors="coerce").fillna(0).astype(int)
    )

    model_result_series = (
        data["model_result"] if "model_result" in data.columns else pd.Series("", index=data.index)
    )
    data["model_result"] = model_result_series.astype(str).str.upper()

    played = data.loc[data["played"] == 1].copy()
    if played.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "games",
                "wins",
                "losses",
                "pushes",
                "hit_rate",
                "qualified_games",
                "qualified_wins",
                "qualified_losses",
                "qualified_pushes",
                "qualified_hit_rate",
            ]
        )

    rows = []

    def _summarize(group: pd.DataFrame, label: str | int) -> dict:
        wins = int((group["model_result"] == "CORRECT").sum())
        losses = int((group["model_result"] == "INCORRECT").sum())
        pushes = int((group["model_result"] == "P").sum())
        total = wins + losses
        hit_rate = float(wins / total) if total else np.nan

        q_mask = group["qualified_edge_flag"] == 1
        q_rows = group.loc[q_mask]
        q_wins = int((q_rows["model_result"] == "CORRECT").sum())
        q_losses = int((q_rows["model_result"] == "INCORRECT").sum())
        q_pushes = int((q_rows["model_result"] == "P").sum())
        q_total = q_wins + q_losses
        q_hit = float(q_wins / q_total) if q_total else np.nan

        return {
            "season": int(year),
            "week": label,
            "games": int(len(group)),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "hit_rate": round(hit_rate, 4) if np.isfinite(hit_rate) else np.nan,
            "qualified_games": int(len(q_rows)),
            "qualified_wins": q_wins,
            "qualified_losses": q_losses,
            "qualified_pushes": q_pushes,
            "qualified_hit_rate": round(q_hit, 4) if np.isfinite(q_hit) else np.nan,
        }

    for wk, grp in played.groupby("week", dropna=True):
        if pd.isna(wk):
            continue
        rows.append(_summarize(grp, int(wk)))

    rows.sort(key=lambda r: (r["week"] == "ALL", r["week"]))
    overall = _summarize(played, "ALL")
    rows.append(overall)

    return pd.DataFrame(rows)


def _load_cached_dataset(name: str) -> pd.DataFrame | None:
    try:
        df = read_dataset(name)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        return None
    return None


def build_backtest_dataset(
    year: int,
    apis: CfbdClients,
    cache: ApiCache,
    *,
    offline: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Building backtest dataset for season %s", year)

    teams_df: pd.DataFrame | None = None
    schedule_df: pd.DataFrame | None = None
    markets_df: pd.DataFrame | None = None

    if offline:
        teams_df = _load_cached_dataset(f"backtest_team_inputs_{year}")
        schedule_df = _load_cached_dataset(f"backtest_schedule_{year}")
        markets_df = _load_cached_dataset(f"backtest_markets_{year}")

    if teams_df is None or teams_df.empty:
        if offline:
            raise RuntimeError(
                f"Cached backtest team inputs for {year} not found. Run online hydration first."
            )
        teams_df = build_team_inputs_datadriven(year, apis, cache)
        write_dataset(teams_df, f"backtest_team_inputs_{year}")

    if schedule_df is None or schedule_df.empty:
        if offline:
            raise RuntimeError(
                f"Cached backtest schedule for {year} not found. Run online hydration first."
            )
        schedule_df = load_schedule_for_year(year, apis, cache)
        if schedule_df is None or schedule_df.empty:
            logger.warning("Backtest schedule empty for %s", year)
            empty = pd.DataFrame()
            write_dataset(empty, f"upa_predictions_{year}_backtest")
            write_dataset(empty, f"backtest_summary_{year}")
            return empty, empty
        write_dataset(schedule_df, f"backtest_schedule_{year}")

    try:
        max_week = int(pd.to_numeric(schedule_df.get("week"), errors="coerce").max())
    except Exception:
        max_week = None
    if not max_week or max_week < 1:
        max_week = 1

    if markets_df is None or markets_df.empty:
        if offline:
            raise RuntimeError(
                f"Cached backtest markets for {year} not found. Run online hydration first."
            )
        markets_df = get_market_lines_for_current_week(year, max_week, schedule_df, apis, cache)
        write_dataset(markets_df, f"backtest_markets_{year}")

    preds_df = build_predictions_for_year(
        year,
        schedule_df,
        apis=apis,
        cache=cache,
        markets_df=markets_df,
        team_inputs_df=teams_df,
    )

    preds_df = _mirror_book_to_legacy_columns(preds_df.copy())
    preds_df = _apply_book_grades(preds_df)

    preds_df["season"] = int(year)
    preds_df["backtest_generated_at"] = pd.Timestamp.utcnow().isoformat()

    summary_df = _compute_backtest_summary(preds_df, year)

    write_dataset(preds_df, f"upa_predictions_{year}_backtest")
    write_dataset(summary_df, f"backtest_summary_{year}")

    logger.info(
        "Backtest build complete: predictions=%s rows, summary=%s rows",
        len(preds_df),
        len(summary_df),
    )

    return preds_df, summary_df


__all__ = ["build_backtest_dataset"]
