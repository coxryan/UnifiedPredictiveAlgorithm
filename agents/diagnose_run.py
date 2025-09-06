# agents/diagnose_run.py
# Verify required CSVs exist, are non-empty, and have expected columns.
import os, sys, pandas as pd

DATA_DIR = "data"

def req(p):
    fp = os.path.join(DATA_DIR, p)
    if not os.path.exists(fp):
        print(f"[FAIL] Missing file: {fp}", file=sys.stderr)
        sys.exit(2)
    return fp

def head(df, n=5, label="sample"):
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(f"[HEAD] {label} ({min(n,len(df))} rows):")
        print(df.head(n).to_string(index=False))

def need_cols(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print(f"[FAIL] {name} missing columns: {miss}", file=sys.stderr)
        print(f"[INFO] {name} columns present: {list(df.columns)}")
        sys.exit(3)

def nonempty(df, name):
    if df.shape[0] == 0:
        print(f"[FAIL] {name} has zero rows", file=sys.stderr)
        sys.exit(4)

def main():
    # team inputs
    ti = pd.read_csv(req("upa_team_inputs_datadriven_v0.csv"))
    print(f"[OK] team inputs rows={ti.shape[0]}")
    need_cols(ti, ["team","wrps_percent_0_100","talent_score_0_100"], "team inputs")
    nonempty(ti, "team inputs")
    head(ti, label="team inputs")

    # schedule
    sch = pd.read_csv(req("cfb_schedule.csv"))
    print(f"[OK] schedule rows={sch.shape[0]}")
    need_cols(sch, ["week","date","away_team","home_team","neutral_site","market_spread_book"], "schedule")
    nonempty(sch, "schedule")
    head(sch[sch["market_spread_book"].astype(str).str.len() > 0], label="schedule (with market)")

    # predictions
    pr = pd.read_csv(req("upa_predictions.csv"))
    print(f"[OK] predictions rows={pr.shape[0]}")
    need_cols(pr, [
        "week","date","away_team","home_team","neutral_site",
        "model_spread_book","market_spread_book","expected_market_spread_book",
        "edge_points_book","value_points_book","qualified_edge_flag",
    ], "predictions")
    nonempty(pr, "predictions")
    head(pr, label="predictions")

    # live edge
    le = pd.read_csv(req("live_edge_report.csv"))
    print(f"[OK] live edge rows={le.shape[0]}")
    need_cols(le, [
        "week","date","away_team","home_team","neutral_site",
        "model_spread_book","market_spread_book","expected_market_spread_book",
        "edge_points_book","value_points_book","qualified_edge_flag",
    ], "live edge")
    nonempty(le, "live edge")
    head(le, label="live edge")

    print("[SUCCESS] Diagnostics passed. All required files present & non-empty.")

if __name__ == "__main__":
    main()