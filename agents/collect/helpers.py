from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        base = os.path.basename(path)
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

