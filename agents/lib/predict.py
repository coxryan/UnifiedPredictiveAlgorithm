import math, pandas as pd

def _safe_float(x):
    try: return float(x)
    except: return float("nan")

def team_advantage_score(row):
    w=_safe_float(row.get("wrps_percent_0_100"))
    t=_safe_float(row.get("talent_score_0_100"))
    p=_safe_float(row.get("portal_net_0_100"))
    return 0.5*(w if math.isfinite(w) else 0)+0.35*(t if math.isfinite(t) else 0)+0.15*(p if math.isfinite(p) else 0)

def home_field_advantage(neutral): return 0.0 if str(neutral)=="1" else 2.2

def model_spread_home(away,home,neutral):
    return round((home-away)*0.15-home_field_advantage(neutral),1)

def expected_market_from_model(model,market):
    if not math.isfinite(model) or not math.isfinite(market): return float("nan")
    delta=model-market; correction=max(-3,min(3,delta)); return model-correction

def value_points(market,expected):
    if not math.isfinite(market) or not math.isfinite(expected): return float("nan")
    return market-expected

def build_predictions_for_year(year,team_inputs,schedule,edge_min=2,value_min=1):
    tmap=team_inputs.set_index("team",drop=False)
    def rating(t): return float(team_advantage_score(tmap.loc[t])) if t in tmap.index else 50.0

    rows=[]
    for _,g in schedule.iterrows():
        home,away=g.get("home_team",""),g.get("away_team","")
        neutral=g.get("neutral_site","0")
        model=model_spread_home(rating(away),rating(home),neutral)
        market=_safe_float(g.get("market_spread_book"))
        rec=g.to_dict(); rec["model_spread_book"]=model

        if math.isfinite(market):
            exp=expected_market_from_model(model,market)
            edge=model-market; val=value_points(market,exp); qual=""
            if math.isfinite(edge) and math.isfinite(val):
                if abs(edge)>=edge_min and abs(val)>=value_min and math.copysign(1,model or 0)==math.copysign(1,exp or 0): qual="1"
            rec.update({"market_spread_book":round(market,1),"expected_market_spread_book":round(exp,1),
                        "edge_points_book":round(edge,1),"value_points_book":round(val,1),"qualified_edge_flag":qual})
        else:
            rec.update({"expected_market_spread_book":"","edge_points_book":"","value_points_book":"","qualified_edge_flag":""})
        rows.append(rec)

    df=pd.DataFrame(rows)
    order=["week","date","away_team","home_team","neutral_site","model_spread_book",
           "market_spread_book","expected_market_spread_book","edge_points_book","value_points_book","qualified_edge_flag"]
    return df[[c for c in order if c in df.columns]+[c for c in df.columns if c not in order]].sort_values(["week","date","away_team","home_team"],ignore_index=True)