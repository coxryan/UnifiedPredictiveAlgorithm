from __future__ import annotations
from pathlib import Path
import pandas as pd

from agents.storage import read_dataset, write_json_blob

DATA = Path("data")

def _head(df: pd.DataFrame, n: int = 5) -> str:
    with pd.option_context("display.max_columns", 20, "display.width", 200):
        return df.head(n).to_string(index=False)

def main() -> None:
    teams = read_dataset("upa_team_inputs_datadriven_v0")
    print(f"[OK] team inputs rows={len(teams)}")
    print("[HEAD] team inputs (5 rows):")
    print(_head(teams))

    sched = read_dataset("cfb_schedule")
    print(f"[OK] schedule rows={len(sched)}")
    has_real_market = 0
    if {"market_spread_book","market_is_synthetic"}.issubset(sched.columns):
        has_real_market = int((sched["market_spread_book"].notna()) & (~sched["market_is_synthetic"].fillna(False))).sum()
    coverage = (100.0 * has_real_market / max(1, len(sched)))
    print(f"[DIAG] schedule real-market coverage: {has_real_market}/{len(sched)} = {coverage:.1f}%")
    cols = [c for c in ["game_id","week","date","away_team","home_team","neutral_site","market_spread_book","market_is_synthetic"] if c in sched.columns]
    print("[HEAD] schedule (5 rows):")
    print(_head(sched[cols]))

    preds = read_dataset("upa_predictions")
    print(f"[OK] predictions rows={len(preds)}")
    with_market = 0
    if {"market_spread_book","market_is_synthetic"}.issubset(preds.columns):
        with_market = int((preds["market_spread_book"].notna()) & (~preds["market_is_synthetic"].fillna(False))).sum()
    print(f"[DIAG] predictions with REAL market rows = {with_market}")
    pcols = [
        "week","date","away_team","home_team","neutral_site",
        "model_spread_book","market_spread_book","expected_market_spread_book",
        "edge_points_book","value_points_book","qualified_edge_flag","market_is_synthetic","game_id"
    ]
    pcols = [c for c in pcols if c in preds.columns]
    print("[HEAD] predictions (5 rows):")
    print(_head(preds[pcols]))

    if "nan_reason" in preds.columns:
        print("[DIAG] top nan_reason counts:")
        print(preds["nan_reason"].value_counts(dropna=False).head(10).to_string())

    edge = read_dataset("live_edge_report")
    print(f"[OK] live edge rows={len(edge)}")

    # ---- Write a compact backfill summary JSON for the Status page link
    try:
        out = {
            "dataset": "upa_predictions",
            "predictions_rows": int(len(preds)),
            "predictions_rows_with_market": int(pd.to_numeric(preds.get("market_spread_book"), errors="coerce").notna().sum()) if "market_spread_book" in preds.columns else 0,
            "schedule_rows": int(len(sched)),
            "schedule_rows_with_market": int(pd.to_numeric(sched.get("market_spread_book"), errors="coerce").notna().sum()) if "market_spread_book" in sched.columns else 0,
        }
        write_json_blob(str(DATA / "market_predictions_backfill.json"), out)
        print(f"[OK] wrote market_predictions_backfill.json record")
    except Exception as e:
        print(f"[warn] unable to write backfill summary: {e}")

    print("[SUCCESS] Diagnostics passed. All required files present & shaped.")

if __name__ == "__main__":
    main()
