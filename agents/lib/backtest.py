# agents/lib/backtest.py
import math, os
import pandas as pd
from .cache import ApiCache, cached_call

def _safe_float(x):
    try: return float(x)
    except: return float("nan")

def iso_date(dt_str):
    return str(dt_str)[:10] if dt_str else ""

def fbs_vs_fbs(g) -> bool:
    return bool(getattr(g,"home_conference",None)) and bool(getattr(g,"away_conference",None))

def bet_result_value_side(home_points: float, away_points: float, market_home: float, model_home: float):
    hp = _safe_float(home_points); ap = _safe_float(away_points)
    if not (math.isfinite(hp) and math.isfinite(ap)): return "", ""
    mkt = _safe_float(market_home); mdl = _safe_float(model_home)
    if not (math.isfinite(mkt) and math.isfinite(mdl)): return "", ""
    edge = mdl - mkt
    if abs(edge) < 1e-9: return "", ""
    if edge > 0:
        delta = (ap - hp) + (-mkt); side = "AWAY"
    else:
        delta = (hp - ap) + (mkt);  side = "HOME"
    if abs(delta) < 1e-9: return "PUSH", side
    return ("CORRECT" if delta > 0 else "INCORRECT"), side

def run_backtest(year_bt: int, team_inputs_bt: pd.DataFrame, build_predictions_fn, apis: dict, cache: ApiCache,
                 data_dir="data", ttl_games=31536000):
    from datetime import datetime
    games_api = apis["games"]
    out_dir = os.path.join(data_dir, str(year_bt))
    os.makedirs(out_dir, exist_ok=True)

    preds = build_predictions_fn(year_bt, team_inputs_bt)

    games, _ = cached_call(cache, "games", {"fn":"get_games","year":year_bt,"season_type":"both"}, ttl_games,
                           lambda: games_api.get_games(year=year_bt, season_type="both"))

    finals=[]
    for g in games or []:
        if not fbs_vs_fbs(g): continue
        hp = getattr(g,"home_points",None); ap = getattr(g,"away_points",None)
        if hp is None or ap is None: continue
        finals.append({
            "week": getattr(g,"week",None),
            "date": iso_date(getattr(g,"start_date",None) or getattr(g,"start_time",None)),
            "home_team": getattr(g,"home_team",""),
            "away_team": getattr(g,"away_team",""),
            "home_points": hp, "away_points": ap
        })
    finals_df = pd.DataFrame(finals)

    df = preds.merge(finals_df, on=["week","date","home_team","away_team"], how="left")
    res = df.apply(lambda r: bet_result_value_side(
        r.get("home_points"), r.get("away_points"),
        _safe_float(r.get("market_spread_book")), _safe_float(r.get("model_spread_book"))
    ), axis=1, result_type="expand")
    df["bet_result_value"] = res[0]; df["bet_side_value"]=res[1]

    p1 = os.path.join(out_dir, "upa_predictions_2024_backtest.csv")
    p2 = os.path.join(out_dir, "backtest_predictions_2024.csv")
    df.to_csv(p1, index=False); df.to_csv(p2, index=False)

    recs=[]
    for wk, grp in df.groupby("week"):
        w = (grp["bet_result_value"]=="CORRECT").sum()
        l = (grp["bet_result_value"]=="INCORRECT").sum()
        p = (grp["bet_result_value"]=="PUSH").sum()
        tot = w + l
        hit = round((w/tot)*100,1) if tot else None
        recs.append({"week": int(wk), "wins": int(w), "losses": int(l), "pushes": int(p), "hit_pct": hit})
    pd.DataFrame(recs).sort_values("week").to_csv(os.path.join(out_dir,"backtest_summary_2024.csv"), index=False)

    print(f"[backtest {year_bt}] wrote: {p1}, {p2}, backtest_summary_2024.csv")