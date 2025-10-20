from __future__ import annotations

from typing import Any, Optional, Dict, Tuple

import logging
import numpy as np
import pandas as pd
import math

from agents.storage import (
    write_dataset as storage_write_dataset,
    read_dataset,
    write_json_blob,
)

logger = logging.getLogger(__name__)


def write_dataset(df: pd.DataFrame, dataset: str) -> None:
    try:
        # Backfill market_spread_book for predictions if missing/empty by joining artifacts we already write.
        # This uses FanDuel-derived markets as the PRIMARY source – it does NOT ignore FanDuel.
        try:
            if dataset == "upa_predictions":
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
                    "write_dataset: initial market gaps=%s dataset=%s",
                    int(missing_mask.sum()),
                    dataset,
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
                        "write_dataset: filled %s rows via %s join=%s for dataset=%s",
                        filled,
                        label,
                        join_cols,
                        dataset,
                    )

                # Step 2: backfill from market_debug snapshot (FanDuel)
                if missing_mask.any():
                    mdbg = read_dataset("market_debug")
                    if not mdbg.empty:
                        if "market_spread_book" not in mdbg.columns and "spread" in mdbg.columns:
                            mdbg = mdbg.rename(columns={"spread": "market_spread_book"})
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
                    sched = read_dataset("cfb_schedule")
                    if not sched.empty:
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
                        "dataset": "upa_predictions",
                        "backfill_before_rows_with_market": n_before,
                        "backfill_after_rows_with_market": n_after,
                        "source": "FanDuel (market_debug) primary; schedule fallback if available",
                    }
                    write_json_blob("market_predictions_backfill", dbg)
                except Exception:
                    pass
                logger.debug(
                    "write_dataset: completed market backfill dataset=%s before=%s after=%s synthetic_rows=%s",
                    dataset,
                    n_before,
                    n_after,
                    int(df.get("market_is_synthetic", pd.Series(dtype=int)).sum()) if "market_is_synthetic" in df.columns else None,
                )
        except Exception:
            # Never fail writing because of backfill logic
            pass
        if dataset in {"upa_predictions", "backtest_predictions_2024", "upa_predictions_2024_backtest"}:
            try:
                df = _mirror_book_to_legacy_columns(df.copy())
            except Exception:
                pass
            try:
                df = _apply_book_grades(df.copy())
            except Exception:
                pass
            try:
                df = _apply_confidence_calibration(df.copy())
            except Exception:
                logger.exception("write_dataset: confidence calibration failed")
    except Exception:
        pass
    storage_write_dataset(df, dataset)


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


def _apply_confidence_calibration(df: pd.DataFrame) -> pd.DataFrame:
    if "model_confidence" not in df.columns:
        return df

    try:
        history = read_dataset("upa_predictions")
    except Exception:
        history = pd.DataFrame()

    conf_bins = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
    spread_bins = [0.0, 3.0, 5.0, 10.0, 15.0, 20.0, float("inf")]
    spread_labels = ["<3", "3-5", "5-10", "10-15", "15-20", "20+"]

    # Prepare historical calibration data (weeks > 3, completed games only)
    calib = pd.DataFrame()
    if history is not None and not history.empty:
        calib = history.copy()
    if not calib.empty:
        for col in ("week", "model_confidence", "market_spread_book"):
            if col in calib.columns:
                calib[col] = pd.to_numeric(calib[col], errors="coerce")
        calib = calib[
            (calib.get("played") == 1)
            & calib.get("week").gt(3)
            & calib["model_result"].isin(["CORRECT", "INCORRECT"])
            & calib["model_confidence"].notna()
        ].copy()
    summary_blob: Dict[str, Any] = {
        "source_rows": int(len(calib)) if not calib.empty else 0,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }

    # Confidence bucket lower bounds (Wilson)
    conf_lower: Dict[Tuple[int, str], float] = {}
    if not calib.empty:
        calib["conf_bucket"] = pd.cut(calib["model_confidence"], bins=conf_bins, include_lowest=True)
        grouped = calib.groupby(["qualified_edge_flag", "conf_bucket"], observed=False)
        conf_stats = []
        for (qualified, bucket), g in grouped:
            wins = (g["model_result"] == "CORRECT").sum()
            losses = (g["model_result"] == "INCORRECT").sum()
            total = wins + losses
            if total == 0:
                continue
            lower, upper = _wilson_bounds(wins, total)
            key = (int(qualified), str(bucket))
            conf_lower[key] = lower
            conf_stats.append(
                {
                    "qualified": int(qualified),
                    "bucket": str(bucket),
                    "bets": int(len(g)),
                    "wins": int(wins),
                    "losses": int(losses),
                    "accuracy": wins / total if total else float("nan"),
                    "lower80": lower,
                    "upper80": upper,
                }
            )
        summary_blob["confidence_stats"] = conf_stats

    # Spread band adjustment (qualified edges only)
    band_adjust: Dict[str, float] = {}
    if not calib.empty:
        calib["band"] = pd.cut(
            calib["market_spread_book"].abs(), bins=spread_bins, labels=spread_labels, right=False
        )
        qualified_hist = calib[calib["qualified_edge_flag"] == 1].copy()
        qualified_hist = qualified_hist.dropna(subset=["band"])
        wins = (qualified_hist["model_result"] == "CORRECT").sum()
        losses = (qualified_hist["model_result"] == "INCORRECT").sum()
        overall_total = wins + losses
        overall_acc = wins / overall_total if overall_total else float("nan")
        summary_blob["overall_accuracy_qualified"] = overall_acc

        band_stats = []
        if not qualified_hist.empty:
            for label, g in qualified_hist.groupby("band", observed=False):
                wins = (g["model_result"] == "CORRECT").sum()
                losses = (g["model_result"] == "INCORRECT").sum()
                total = wins + losses
                if total == 0:
                    continue
                acc = wins / total
                adj = acc - overall_acc if pd.notna(overall_acc) else 0.0
                adj = max(min(adj, 0.05), -0.05)
                band_adjust[str(label)] = adj
                lower, upper = _wilson_bounds(wins, total)
                band_stats.append(
                    {
                        "band": str(label),
                        "bets": int(len(g)),
                        "wins": int(wins),
                        "losses": int(losses),
                        "accuracy": acc,
                        "lower80": lower,
                        "upper80": upper,
                        "adj": adj,
                    }
                )
        summary_blob["band_stats"] = band_stats

    # Apply calibration to current frame
    if "market_spread_book" in df.columns:
        df["__abs_band__"] = pd.cut(
            pd.to_numeric(df["market_spread_book"], errors="coerce").abs(),
            bins=spread_bins,
            labels=spread_labels,
            right=False,
        )
    else:
        df["__abs_band__"] = pd.Series([None] * len(df))

    df["__conf_bucket__"] = pd.cut(df["model_confidence"], bins=conf_bins, include_lowest=True)

    calibrated = []
    for idx, row in df.iterrows():
        base = row.get("model_confidence", float("nan"))
        bucket = str(row.get("__conf_bucket__"))
        qualified = int(row.get("qualified_edge_flag") or 0)
        band = str(row.get("__abs_band__"))

        if (qualified, bucket) in conf_lower and pd.notna(conf_lower[(qualified, bucket)]):
            base = conf_lower[(qualified, bucket)]

        if pd.isna(base):
            base = row.get("model_confidence", 0.6)

        adj = band_adjust.get(band, 0.0)

        # Penalize weak historical band
        if band == "15-20":
            adj -= 0.03

        # Reward consistent 20+ band slightly (within clamp)
        if band == "20+":
            adj += 0.02

        calibrated_prob = base + adj

        source = str(row.get("market_spread_source", "")).strip().lower()
        if source not in {"fanduel", ""}:
            calibrated_prob -= 0.03

        # Guardrails
        calibrated_prob = max(0.45, min(0.80, calibrated_prob))

        calibrated.append(calibrated_prob)

    df["confidence_calibrated"] = pd.Series(calibrated, index=df.index)
    df["confidence_bucket"] = df["__conf_bucket__"].astype(str)
    df["confidence_band"] = df["__abs_band__"].astype(str)
    df["confidence_play_flag"] = (
        (df["confidence_calibrated"] >= 0.62) & (df.get("qualified_edge_flag", 0) == 1)
    ).astype(int)

    df.drop(columns=["__conf_bucket__", "__abs_band__"], inplace=True)

    try:
        write_json_blob("confidence_calibration_stats", summary_blob)
    except Exception:
        pass

    return df


def _wilson_bounds(wins: int, total: int, z: float = 1.2815515655446004) -> Tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = wins / total
    denom = 1 + z**2 / total
    center = p + z**2 / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return lower, upper


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
    "write_dataset",
    "_safe_float",
    "_grade_pick_result",
    "_apply_book_grades",
    "_mirror_book_to_legacy_columns",
    "_normalize_percent",
    "_scale_0_100",
]
