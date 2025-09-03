# agents/lib/predict.py
import math
import pandas as pd

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def team_advantage_score(row: pd.Series) -> float:
    w = _safe_float(row.get("wrps_percent_0_100"))
    t = _safe_float(row.get("talent_score_0_100"))
    p = _safe_float(row.get("portal_net_0_100"))
    w = 0.0 if not math.isfinite(w) else w
    t = 0.0 if not math.isfinite(t) else t
    p = 0.0 if not math.isfinite(p) else p
    return 0.50 * w + 0.35 * t + 0.15 * p

def home_field_advantage(neutral_flag: str) -> float:
    return 0.0 if str(neutral_flag) == "1" else 2.2

def model_spread_home(away_rating: float, home_rating: float, neutral_flag: str) -> float:
    base_diff_pts = (home_rating - away_rating) * 0.15
    hfa = home_field_advantage(neutral_flag)
    return round(base_diff_pts - hfa, 1)

def expected_market_from_model(model_home: float, market_home: float) -> float:
    if not math.isfinite(model_home) or not math.isfinite(market_home):
        return float("nan")
    delta = model_home - market_home
    correction = max(-3.0, min(3.0, delta))
    return model_home - correction

def value_points(market_home: float, expected_home: float) -> float:
    if not (math.isfinite(market_home) and math.isfinite(expected_home)):
        return float("nan")
    return market_home - expected_home

def build_predictions_for_year(
    year: int,
    team_inputs: pd.DataFrame,
    schedule: pd.DataFrame,
    edge_min: float = 2.0,
    value_min: float = 1.0,
) -> pd.DataFrame:
    tmap = team_inputs.set_index("team", drop=False)

    def rating(team: str) -> float:
        return float(team_advantage_score(tmap.loc[team])) if team in tmap.index else 50.0

    rows = []
    for _, g in schedule.iterrows():
        home, away = str(g.get("home_team","")), str(g.get("away_team",""))
        neutral = g.get("neutral_site","0")

        model_home = model_spread_home(rating(away), rating(home), neutral)
        market_home = _safe_float(g.get("market_spread_book"))

        rec = g.to_dict()
        rec["model_spread_book"] = model_home

        if math.isfinite(market_home):
            exp = expected_market_from_model(model_home, market_home)
            edge = model_home - market_home
            val  = value_points(market_home, exp)
            qual = ""
            if math.isfinite(edge) and math.isfinite(val):
                if abs(edge) >= edge_min and abs(val) >= value_min:
                    if math.copysign(1, model_home or 0.0) == math.copysign(1, exp or 0.0):
                        qual = "1"
            rec.update({
                "market_spread_book": round(market_home,1),
                "expected_market_spread_book": round(exp,1),
                "edge_points_book": round(edge,1),
                "value_points_book": round(val,1),
                "qualified_edge_flag": qual,
            })
        else:
            rec.update({
                "expected_market_spread_book": "",
                "edge_points_book": "",
                "value_points_book": "",
                "qualified_edge_flag": "",
            })

        rows.append(rec)

    df = pd.DataFrame(rows)
    order = [
        "week","date","away_team","home_team","neutral_site",
        "model_spread_book","market_spread_book","expected_market_spread_book",
        "edge_points_book","value_points_book","qualified_edge_flag"
    ]
    df = df[[c for c in order if c in df.columns] + [c for c in df.columns if c not in order]]
    return df.sort_values(["week","date","away_team","home_team"], ignore_index=True)