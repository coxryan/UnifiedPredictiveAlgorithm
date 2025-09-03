# agents/lib/team_inputs.py
import math
import pandas as pd
from collections import defaultdict
from .cache import ApiCache, cached_call

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def _normalize_pct(x):
    x = _safe_float(x)
    if not math.isfinite(x): return float("nan")
    return x*100.0 if x <= 1.0 else x

def _scale_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([50.0]*len(s), index=s.index)
    mn, mx = s.min(), s.max()
    if not math.isfinite(mn) or not math.isfinite(mx) or mx == mn:
        return pd.Series([50.0]*len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100.0

EXPECTED_RETURNING_COLS = [
    "team","conference","wrps_offense_percent","wrps_defense_percent","wrps_overall_percent","wrps_percent_0_100"
]

def build_team_inputs(year: int, apis: dict, cache: ApiCache,
                      ttl_fbs=86400, ttl_return=604800, ttl_talent=604800, ttl_portal=86400, ttl_srs=31536000, ttl_games=31536000):
    teams_api   = apis["teams"]
    players_api = apis["players"]
    ratings_api = apis["ratings"]
    games_api   = apis["games"]

    # FBS map
    fbs, _ = cached_call(cache, "teams_fbs", {"fn":"get_fbs_teams","year":year}, ttl_fbs, lambda: teams_api.get_fbs_teams(year=year))
    team_conf = {t.school: (t.conference or "FBS") for t in fbs}
    conferences = sorted({t.conference for t in fbs if t.conference})

    # Returning production
    rows = []
    for conf in conferences:
        items, _ = cached_call(cache, "returning",
                               {"fn":"get_returning_production","year":year,"conference":conf},
                               ttl_return,
                               lambda: players_api.get_returning_production(year=year, conference=conf))
        for it in items or []:
            rows.append({
                "team": getattr(it,"team",None),
                "conference": getattr(it,"conference",None) or team_conf.get(getattr(it,"team",None),"FBS"),
                "_overall": getattr(it, "overall", None),
                "_offense": getattr(it, "offense", None),
                "_defense": getattr(it, "defense", None),
                "_ppa_tot": getattr(it, "total_ppa", None),
                "_ppa_off": getattr(it, "total_offense_ppa", None) or (getattr(it,"total_passing_ppa",0)+getattr(it,"total_rushing_ppa",0)),
                "_ppa_def": getattr(it, "total_defense_ppa", None) or getattr(it, "total_defensive_ppa", None),
            })
    rp = pd.DataFrame(rows).drop_duplicates(subset=["team"])
    rp["wrps_offense_percent"] = rp["_offense"].apply(_normalize_pct)
    rp["wrps_defense_percent"] = rp["_defense"].apply(_normalize_pct)
    rp["wrps_overall_percent"] = rp["_overall"].apply(_normalize_pct)
    need_proxy = (rp["wrps_overall_percent"].isna().all() or rp["wrps_offense_percent"].isna().all() or rp["wrps_defense_percent"].isna().all())
    if need_proxy:
        if rp["wrps_overall_percent"].isna().all(): rp["wrps_overall_percent"] = _scale_0_100(rp["_ppa_tot"])
        if rp["wrps_offense_percent"].isna().all(): rp["wrps_offense_percent"] = _scale_0_100(rp["_ppa_off"])
        if rp["wrps_defense_percent"].isna().all(): rp["wrps_defense_percent"] = _scale_0_100(rp["_ppa_def"])
    rp["wrps_percent_0_100"] = pd.to_numeric(rp["wrps_overall_percent"], errors="coerce").round(1)
    for c in ["team","conference"]+EXPECTED_RETURNING_COLS[2:]:
        if c not in rp.columns: rp[c] = None
    rp["conference"] = rp["team"].map(team_conf).fillna(rp["conference"]).fillna("FBS")
    rp = rp[EXPECTED_RETURNING_COLS].copy()

    # Talent
    items, _ = cached_call(cache, "talent", {"fn":"get_talent","year":year}, ttl_talent, lambda: teams_api.get_talent(year=year))
    tdf = pd.DataFrame([{"team": x.team, "talent": float(getattr(x,"talent",0) or 0)} for x in items or []])
    if tdf.empty:
        tdf["team"]=[]; tdf["talent_score_0_100"]=[]
    else:
        mn, mx = tdf["talent"].min(), tdf["talent"].max()
        tdf["talent_score_0_100"] = 50.0 if mx==mn else ((tdf["talent"]-mn)/(mx-mn)*100.0).round(1)
    tdf = tdf[["team","talent_score_0_100"]]

    # Prior SOS via SRS & games
    srs, _ = cached_call(cache, "srs", {"fn":"get_srs","year":year-1}, ttl_srs, lambda: ratings_api.get_srs(year=year-1))
    srs_map = {x.team: float(x.rating or 0) for x in srs}
    games, _ = cached_call(cache, "games", {"fn":"get_games","year":year-1,"season_type":"both"}, ttl_games, lambda: games_api.get_games(year=year-1, season_type="both"))
    opps = defaultdict(list)
    for g in games:
        ht, at = getattr(g,"home_team",None), getattr(g,"away_team",None)
        if not ht or not at: continue
        if ht in srs_map and at in srs_map:
            opps[ht].append(srs_map[at]); opps[at].append(srs_map[ht])
    sos = pd.DataFrame([{"team":t,"sos_value":sum(v)/len(v)} for t,v in opps.items() if v])
    if sos.empty:
        sos["team"]=[]; sos["prev_season_sos_rank_1_133"]=[]
    else:
        sos["prev_season_sos_rank_1_133"] = sos["sos_value"].rank(ascending=False, method="min").astype(int)
    sos = sos[["team","prev_season_sos_rank_1_133"]]

    # Transfer portal
    portal, _ = cached_call(cache, "portal", {"fn":"get_transfer_portal","year":year}, ttl_portal, lambda: players_api.get_transfer_portal(year=year))
    inc = defaultdict(int); out = defaultdict(int); rin = defaultdict(float); rout = defaultdict(float)
    for p in portal or []:
        to_t = getattr(p,"destination",None) or getattr(p,"to_team",None)
        fr_t = getattr(p,"origin",None) or getattr(p,"from_team",None)
        rating = getattr(p,"rating",None); stars = getattr(p,"stars",None)
        try: val = float(rating) if isinstance(rating,(int,float)) else (float(stars) if isinstance(stars,(int,float)) else 1.0)
        except: val = 1.0
        if to_t: inc[to_t]+=1; rin[to_t]+=val
        if fr_t: out[fr_t]+=1; rout[fr_t]+=val
    rows=[]
    for t in set(list(inc.keys())+list(out.keys())):
        rows.append({"team":t,"portal_net_count":inc[t]-out[t],"portal_net_value":rin[t]-rout[t]})
    pdf = pd.DataFrame(rows)
    if pdf.empty:
        pdf["team"]=[]; pdf["portal_net_0_100"]=[]; pdf["portal_net_count"]=[]; pdf["portal_net_value"]=[]
    else:
        pdf["portal_net_0_100"] = (0.5*_scale_0_100(pdf["portal_net_count"]) + 0.5*_scale_0_100(pdf["portal_net_value"])).round(1)
    pdf = pdf[["team","portal_net_0_100","portal_net_count","portal_net_value"]]

    # Merge
    df = rp.merge(tdf, on="team", how="left") \
           .merge(sos, on="team", how="left") \
           .merge(pdf, on="team", how="left")

    if "conference" in df.columns:
        df.sort_values(["conference","team"], inplace=True, ignore_index=True)
    else:
        df.sort_values(["team"], inplace=True, ignore_index=True)
    return df