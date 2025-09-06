# agents/diagnose_run.py
import os, sys
import pandas as pd

DATA = "data"

def head(df, n=5):
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(n).to_string(index=False))

def require_file(path: str):
    if not os.path.exists(path):
        print(f"[FAIL] Missing file: {path}", file=sys.stderr)
        sys.exit(2)

def require_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[FAIL] {name} missing columns: {missing}", file=sys.stderr)
        print(f"[INFO] {name} columns present: {list(df.columns)}")
        sys.exit(3)

def check_nonempty(df: pd.DataFrame, name: str):
    if df.shape[0] == 0:
        print(f"[FAIL] {name} has zero rows", file=sys.stderr)
        sys.exit(4)

def main():
    # 1) Team inputs
    p_inputs = os.path.join(DATA, "upa_team_inputs_datadriven_v0.csv")
    require_file(p_inputs)
    ti = pd.read_csv(p_inputs)
    print(f"[OK] team inputs rows={ti.shape[0]}")
    require_cols(ti, ["team","wrps_percent_0_100","talent_score_0_100"], "team inputs")
    check_nonempty(ti, "team inputs")
    head(ti)

    # 2) Schedule
    p_sched = os.path.join(DATA, "cfb_schedule.csv")
    require_file(p_sched)
    sch = pd.read_csv(p_sched)
    print(f"[OK] schedule rows={sch.shape[0]}")
    require_cols(sch, ["week","date","away_team","home_team","neutral_site","market_spread_book"], "schedule")
    check_nonempty(sch, "schedule")
    # show current-week sample (where market likely exists)
    print("[INFO] sample rows with market_spread_book present:")
    head(sch[sch["market_spread_book"].astype(str).str.len() > 0])

    # 3) Predictions
    p_pred = os.path.join(DATA, "upa_predictions.csv")
    require_file(p_pred)
    pr = pd.read_csv(p_pred)
    print(f"[OK] predictions rows={pr.shape[0]}")
    require_cols(
        pr,
        ["week","date","away_team","home_team","neutral_site","model_spread_book",
         "market_spread_book","expected_market_spread_book","edge_points_book","value_points_book","qualified_edge_flag"],
        "predictions"
    )
    check_nonempty(pr, "predictions")
    head(pr)

    # 4) Live edge
    p_live = os.path.join(DATA, "live_edge_report.csv")
    require_file(p_live)
    le = pd.read_csv(p_live)
    print(f"[OK] live edge rows={le.shape[0]}")
    require_cols(
        le,
        ["week","date","away_team","home_team","neutral_site","model_spread_book",
         "market_spread_book","expected_market_spread_book","edge_points_book","value_points_book","qualified_edge_flag"],
        "live edge"
    )
    check_nonempty(le, "live edge")
    head(le)

    print("[SUCCESS] Diagnostics passed. All required files present & non-empty.")

if __name__ == "__main__":
    main()