# agents/lib/predict.py
import math
import pandas as pd

def _safe_float(x):
    try: return float(x)
    except: return float("nan")

def team_advantage_score(row: pd.Series) -> float:
    w = _safe_float(row.get("wrps_percent_0_100"))
    t = _safe_float(row.get("talent_score_0_100"))
    p = _safe_float(row.get("portal_net_0_100"))
    w = 0.0 if not math.isfinite(w) else w
    t = 0.0 if not math.isfinite(t) else t
    p = 0.0 if not math.isfinite(p) else p
    return 0.5*w + 0.35*t + 0.15*p

def home_field(neutral_flag: str) -> float:
    return 0.0 if str(neutral_flag)=="1" else 2.2

def expected_market_from_model(model_home: float, market_home: float) -> float:
    if not math.isfinite(model_home) or not math.isfinite(market_home):
        return float("nan")
    delta = model_home - market_home
    correction = max(-3.0, min(3.0, delta))
    return model_home - correction

def value_side_from_edge(edge: float, home_team: str, away_team: str) -> str:
    if not math.isfinite(edge) or abs(edge) < 1e-9: return ""
    return f"{away_team} (away)" if edge > 0 else f"{home_team} (home)"

def build_predictions_for_year(year: int, team_inputs: pd.DataFrame, schedule: pd.DataFrame,
                               edge_min=2.0, value_min=1.0) -> pd.DataFrame:
    tmap = team_inputs.set_index("team", drop=False)

    def team_rating(name: str) -> float:
        if name not in tmap.index: return 50.0
        return float(team_advantage_score(tmap.loc[name]))

    out=[]
    for _, r in schedule.iterrows():
        home = r["home_team"]; away = r["away_team"]
        home_rt = team_rating(home); away_rt = team_rating(away)
        base_diff_pts = (home_rt - away_rt) * 0.15
        hfa = home_field(r["neutral_site"])
        model_home = round(base_diff_pts - hfa, 1)  # negative => home favorite (book-style)

        market_home = _safe_float(r.get("market_spread_book"))
        exp_home = expected_market_from_model(model_home, market_home) if math.isfinite(market_home) else float("nan")
        edge = model_home - market_home if math.isfinite(market_home) else float("nan")
        value = market_home - exp_home if (math.isfinite(market_home) and math.isfinite(exp_home)) else float("nan")

        qual = ""
        if math.isfinite(edge) and math.isfinite(value):
            if abs(edge) >= edge_min and abs(value) >= value_min:
                if math.copysign(1, model_home) == math.copysign(1, (exp_home if math.isfinite(exp_home) else model_home)):
                    qual = "1"

        rec = {**r.to_dict(),
               "model_spread_book": round(model_home,1),
               "expected_market_spread_book": round(exp_home,1) if math.isfinite(exp_home) else "",
               "edge_points_book": round(edge,1) if math.isfinite(edge) else "",
               "value_points_book": round(value,1) if math.isfinite(value) else "",
               "qualified_edge_flag": qual}
        out.append(rec)

    df = pd.DataFrame(out).sort_values(["week","date","away_team","home_team"], ignore_index=True)
    return df