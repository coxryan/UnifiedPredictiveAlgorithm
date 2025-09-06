import os, math, pandas as pd
from .cache import ApiCache
from .predict import build_predictions_for_year
from .market import _iso_date, _fbs_vs_fbs, _line_to_home_perspective

def _write_csv(path,df): os.makedirs(os.path.dirname(path),exist_ok=True); df.to_csv(path,index=False); print(f"Wrote {path} ({df.shape[0]} rows)")

def _fetch_full_season_lines(year,apis):
    games=apis["games"].get_games(year=year,season_type="both") or []
    bet_api=apis["betting"]; recs=[]
    for g in games:
        if not _fbs_vs_fbs(g): continue
        gid,ht,wk=getattr(g,"id",None),getattr(g,"home_team",None),getattr(g,"week",None)
        if not gid or not ht or not wk: continue
        try: lines=bet_api.get_lines(year=year,week=int(wk),team=ht) or []
        except: lines=[]
        vals=[]
        for ln in lines:
            for snap in getattr(ln,"lines",[]) or []:
                v=_line_to_home_perspective(snap)
                if math.isfinite(v): vals.append(v)
        if vals: vals.sort(); recs.append({"game_id":gid,"market_spread_book":round(vals[len(vals)//2],1)})
    return pd.DataFrame(recs).drop_duplicates("game_id")

def run_backtest(year,team_inputs,apis,cache:ApiCache|None=None,data_dir="data"):
    if cache is None: cache=ApiCache(os.path.join(data_dir,".api_cache"))
    games=apis["games"].get_games(year=year,season_type="both") or []
    rows=[{"game_id":getattr(g,"id",None),"week":getattr(g,"week",None),
           "date":_iso_date(getattr(g,"start_date",None) or getattr(g,"start_time",None)),
           "away_team":getattr(g,"away_team",""),"home_team":getattr(g,"home_team",""),
           "neutral_site":"1" if bool(getattr(g,"neutral_site",False)) else "0"} for g in games if _fbs_vs_fbs(g)]
    sched=pd.DataFrame(rows)
    sched=sched.merge(_fetch_full_season_lines(year,apis),on="game_id",how="left")
    preds=build_predictions_for_year(year,team_inputs,sched)
    _write_csv(os.path.join(data_dir,f"backtest_predictions_{year}.csv"),preds)
    summary=pd.DataFrame([{"year":year,"games_with_market":preds["market_spread_book"].notna().sum(),"games_total":len(preds)}])
    _write_csv(os.path.join(data_dir,f"backtest_summary_{year}.csv"),summary)