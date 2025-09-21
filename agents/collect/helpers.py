from __future__ import annotations

import os
from typing import Any, Optional

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        base = os.path.basename(path)
        # Backfill market_spread_book for predictions if missing/empty by joining artifacts we already write.
        # This uses FanDuel-derived market_debug.csv as the PRIMARY source – it does NOT ignore FanDuel.
        try:
            if base == "upa_predictions.csv":
                n_before = int(pd.to_numeric(df.get("market_spread_book"), errors="coerce").notna().sum()) if ("market_spread_book" in df.columns) else 0

                # Normalize keys for downstream joins
                for col in ("game_id", "week"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                for col in ("home_team", "away_team"):
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip()

                if "market_spread_book" not in df.columns:
                    df["market_spread_book"] = np.nan

                def _numeric_series(name: str) -> pd.Series:
                    return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series([], dtype="float64")

                market_numeric = _numeric_series("market_spread_book")
                missing_mask = market_numeric.isna()
                logger.debug(
                    "write_csv: initial market gaps=%s file=%s",
                    int(missing_mask.sum()),
                    path,
                )

                # Step 1: copy from legacy column when available
                if missing_mask.any() and "market_spread" in df.columns:
                    candidate = pd.to_numeric(df["market_spread"], errors="coerce")
                    df.loc[missing_mask, "market_spread_book"] = candidate.loc[missing_mask]
                    market_numeric = _numeric_series("market_spread_book")
                    missing_mask = market_numeric.isna()

                def _fill_from_source(source_df: pd.DataFrame, join_cols: list[str], label: str) -> None:
                    nonlocal market_numeric, missing_mask
                    if not missing_mask.any() or not join_cols:
                        return
                    if "market_spread_book" not in source_df.columns:
                        return
                    if not set(join_cols).issubset(source_df.columns):
                        return
                    idx = df.index[missing_mask]
                    if idx.empty:
                        return
                    prev_missing = int(missing_mask.sum())
                    subset = df.loc[idx, join_cols].copy().reset_index().rename(columns={"index": "__idx"})
                    tmp = source_df[join_cols + ["market_spread_book"]].drop_duplicates(join_cols)
                    merged = subset.merge(tmp, on=join_cols, how="left")
                    df.loc[merged["__idx"], "market_spread_book"] = merged["market_spread_book"].values
                    market_numeric = _numeric_series("market_spread_book")
                    missing_mask = market_numeric.isna()
                    filled = prev_missing - int(missing_mask.sum())
                    logger.debug(
                        "write_csv: filled %s rows via %s join=%s for file=%s",
                        filled,
                        label,
                        join_cols,
                        path,
                    )

                # Step 2: backfill from market_debug snapshot (FanDuel)
                if missing_mask.any():
                    mdbg_p = os.path.join(os.path.dirname(path), "market_debug.csv")
                    if os.path.exists(mdbg_p):
                        mdbg = pd.read_csv(mdbg_p)
                        for col in ("game_id", "week"):
                            if col in mdbg.columns:
                                mdbg[col] = pd.to_numeric(mdbg[col], errors="coerce")
                        for col in ("home_team", "away_team"):
                            if col in mdbg.columns:
                                mdbg[col] = mdbg[col].astype(str).str.strip()

                        _fill_from_source(mdbg, [c for c in ["game_id", "week"] if c in df.columns], "market_debug:game_week")
                        _fill_from_source(mdbg, [c for c in ["home_team", "away_team", "week"] if c in df.columns], "market_debug:matchup")

                # Step 3: fall back to schedule-derived spreads if present
                if missing_mask.any():
                    sched_p = os.path.join(os.path.dirname(path), "cfb_schedule.csv")
                    if os.path.exists(sched_p):
                        sched = pd.read_csv(sched_p)
                        if "market_spread_book" in sched.columns:
                            for col in ("game_id", "week"):
                                if col in sched.columns:
                                    sched[col] = pd.to_numeric(sched[col], errors="coerce")
                            for col in ("home_team", "away_team"):
                                if col in sched.columns:
                                    sched[col] = sched[col].astype(str).str.strip()
                            _fill_from_source(sched, [c for c in ["game_id", "week"] if c in df.columns], "schedule:game_week")
                            _fill_from_source(sched, [c for c in ["home_team", "away_team", "week"] if c in df.columns], "schedule:matchup")

                # Final safety: convert dtype and fall back to model numbers only where still empty
                if "market_spread_book" in df.columns:
                    df["market_spread_book"] = pd.to_numeric(df["market_spread_book"], errors="coerce")
                    market_numeric = df["market_spread_book"]
                    missing_mask = market_numeric.isna()

                if missing_mask.any() and "model_spread_book" in df.columns:
                    model_vals = pd.to_numeric(df["model_spread_book"], errors="coerce")
                    df.loc[missing_mask, "market_spread_book"] = model_vals.loc[missing_mask]
                    if "market_is_synthetic" in df.columns:
                        df.loc[:, "market_is_synthetic"] = df["market_is_synthetic"].astype(int)
                        df.loc[missing_mask, "market_is_synthetic"] = 1
                        df.loc[~missing_mask, "market_is_synthetic"] = 0
                else:
                    if "market_is_synthetic" in df.columns:
                        df.loc[:, "market_is_synthetic"] = df["market_is_synthetic"].astype(int)
                        df.loc[missing_mask, "market_is_synthetic"] = 1
                        df.loc[~missing_mask, "market_is_synthetic"] = 0

                # Emit a tiny debug summary so we can verify FanDuel spreads are landing
                try:
                    n_after = int(pd.to_numeric(df.get("market_spread_book"), errors="coerce").notna().sum()) if ("market_spread_book" in df.columns) else 0
                    dbg = {
                        "file": "upa_predictions.csv",
                        "backfill_before_rows_with_market": n_before,
                        "backfill_after_rows_with_market": n_after,
                        "source": "FanDuel (market_debug.csv) primary; schedule fallback if available",
                    }
                    with open(os.path.join(os.path.dirname(path), "market_predictions_backfill.json"), "w") as f:
                        import json as _json
                        _json.dump(dbg, f, indent=2)
                except Exception:
                    pass
                logger.debug(
                    "write_csv: completed market backfill file=%s before=%s after=%s synthetic_rows=%s",
                    path,
                    n_before,
                    n_after,
                    int(df.get("market_is_synthetic", pd.Series(dtype=int)).sum()) if "market_is_synthetic" in df.columns else None,
                )
        except Exception:
            # Never fail writing because of backfill logic
            pass
        if base in ("upa_predictions.csv", "backtest_predictions_2024.csv", "upa_predictions_2024_backtest.csv"):
            try:
                df = _mirror_book_to_legacy_columns(df.copy())
            except Exception:
                pass
            try:
                df = _apply_book_grades(df.copy())
            except Exception:
                pass
    except Exception:
        pass
    df.to_csv(path, index=False)


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _grade_pick_result(pick_side, home_points, away_points, market_home_line) -> str:
    """Grade book-style pick result (CORRECT/INCORRECT/P or empty if not gradeable)."""
    try:
        if pick_side is None or str(pick_side).strip() == "":
            return ""
        hp = float(home_points) if home_points is not None else float("nan")
        ap = float(away_points) if away_points is not None else float("nan")
        m = float(market_home_line) if market_home_line is not None else float("nan")
        if not (np.isfinite(hp) and np.isfinite(ap) and np.isfinite(m)):
            return ""
        adj = (hp - ap) + m  # home covers if positive
        if abs(adj) < 1e-9:
            return "P"
        cover_home = 1 if adj > 0 else -1
        ps = str(pick_side).upper()
        pick_home = ("HOME" in ps) or ("(HOME)" in ps)
        pick_away = ("AWAY" in ps) or ("(AWAY)" in ps)
        if pick_home:
            return "CORRECT" if cover_home > 0 else "INCORRECT"
        if pick_away:
            return "CORRECT" if cover_home < 0 else "INCORRECT"
        return ""
    except Exception:
        return ""


def _apply_book_grades(df: pd.DataFrame) -> pd.DataFrame:
    req = {"home_points", "away_points", "market_spread_book"}
    if not req.issubset(df.columns):
        return df

    for c in [
        "home_points",
        "away_points",
        "market_spread_book",
        "edge_points_book",
        "model_spread_book",
        "expected_market_spread_book",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _norm_pick(s):
        if s is None:
            return ""
        t = str(s).strip().upper()
        if "AWAY" in t:
            return "AWAY"
        if "HOME" in t:
            return "HOME"
        return ""

    if "model_pick_side" in df.columns:
        model_pick = df["model_pick_side"].map(_norm_pick)
    else:
        edge = df.get("edge_points_book")
        if edge is None:
            edge = pd.to_numeric(df.get("model_spread_book"), errors="coerce") - pd.to_numeric(
                df.get("market_spread_book"), errors="coerce"
            )
        edge = pd.to_numeric(edge, errors="coerce")
        model_pick = edge.apply(lambda e: "AWAY" if pd.notna(e) and e > 0 else ("HOME" if pd.notna(e) else ""))

    expected_pick = (
        df["expected_pick_side"].map(_norm_pick)
        if "expected_pick_side" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )

    df["model_result"] = [
        _grade_pick_result(p, hp, ap, m)
        for p, hp, ap, m in zip(model_pick, df["home_points"], df["away_points"], df["market_spread_book"])
    ]
    df["expected_result"] = [
        _grade_pick_result(p, hp, ap, m)
        for p, hp, ap, m in zip(expected_pick, df["home_points"], df["away_points"], df["market_spread_book"])
    ]
    return df


def _mirror_book_to_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror FanDuel/“book” columns into legacy columns the UI expects."""
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df

        def _missing(col: str) -> bool:
            return (col not in df.columns) or pd.to_numeric(df[col], errors="coerce").isna().all()

        mappings = [
            ("market_spread_book", "market_spread"),
            ("market_spread_book", "market_h"),
            ("model_spread_book", "model_spread"),
            ("edge_points_book", "edge"),
            ("value_points_book", "value"),
            ("ev_percent_book", "ev_percent"),
            ("ev_bps_book", "ev_bps"),
            ("expected_market_spread_book", "expected_market_spread"),
        ]
        for src, dst in mappings:
            if src in df.columns and _missing(dst):
                df[dst] = df[src]
        return df
    except Exception:
        return df


def _normalize_percent(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return x * 100.0 if x <= 1.0 else x


def _scale_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series(50.0, index=s.index, dtype="float64")
    out = (s - mn) / (mx - mn) * 100.0
    return out.astype("float64")


__all__ = [
    "write_csv",
    "_safe_float",
    "_grade_pick_result",
    "_apply_book_grades",
    "_mirror_book_to_legacy_columns",
    "_normalize_percent",
    "_scale_0_100",
]
