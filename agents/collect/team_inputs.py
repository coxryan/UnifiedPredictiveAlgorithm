from __future__ import annotations

import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .cfbd_clients import CfbdClients
from .cache import ApiCache
from .helpers import _normalize_percent, _scale_0_100
from agents.storage import read_dataset, write_dataset as storage_write_dataset, delete_rows


def build_team_inputs_datadriven(year: int, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    # conference map
    team_conf: Dict[str, str] = {}
    teams_table = read_dataset("raw_cfbd_fbs_teams")
    if not teams_table.empty:
        tbl = teams_table.loc[teams_table.get("year") == year]
        if not tbl.empty:
            team_conf = {
                str(row["team"]): (row.get("conference") or "FBS")
                for _, row in tbl.iterrows()
                if row.get("team")
            }
    if not team_conf and apis.teams_api:
        try:
            fbs = apis.teams_api.get_fbs_teams(year=year)
            team_conf = {t.school: (t.conference or "FBS") for t in (fbs or []) if getattr(t, "school", None)}
            if team_conf:
                df_store = pd.DataFrame(
                    [
                        {
                            "team": getattr(t, "school", None),
                            "conference": getattr(t, "conference", None),
                            "year": year,
                            "team_id": getattr(t, "id", None),
                        }
                        for t in (fbs or [])
                        if getattr(t, "school", None)
                    ]
                )
                delete_rows("raw_cfbd_fbs_teams", "year", year)
                storage_write_dataset(df_store, "raw_cfbd_fbs_teams", if_exists="append")
        except Exception as e:
            print(f"[warn] fbs teams fetch failed: {e}", file=sys.stderr)

    # Returning Production (Connelly via CFBD)
    rp_df = read_dataset("raw_cfbd_returning_production")
    if not rp_df.empty:
        rp_df = rp_df.loc[rp_df.get("year") == year].copy()
    if rp_df.empty and apis.players_api and team_conf:
        rp_rows: List[Dict[str, Any]] = []
        conferences = sorted({c for c in team_conf.values() if c})
        for conf in conferences:
            key = f"rp:{year}:{conf}"
            ok, data = cache.get(key)
            if ok:
                items = data
            else:
                try:
                    items = apis.players_api.get_returning_production(year=year, conference=conf)
                    serial = []
                    for it in items or []:
                        serial.append(
                            {
                                "team": getattr(it, "team", None),
                                "conference": getattr(it, "conference", None),
                                "overall": getattr(it, "overall", None),
                                "offense": getattr(it, "offense", None),
                                "defense": getattr(it, "defense", None),
                                "total_ppa": getattr(it, "total_ppa", None),
                                "total_offense_ppa": getattr(it, "total_offense_ppa", None),
                                "total_defense_ppa": getattr(it, "total_defense_ppa", None)
                                or getattr(it, "total_defensive_ppa", None),
                                "total_passing_ppa": getattr(it, "total_passing_ppa", None),
                                "total_rushing_ppa": getattr(it, "total_rushing_ppa", None),
                            }
                        )
                    cache.set(key, serial)
                    items = serial
                except Exception as e:
                    print(f"[warn] returning production fetch failed for {conf}: {e}", file=sys.stderr)
                    items = []

            for it in items or []:
                rp_rows.append(
                    {
                        "team": it.get("team"),
                        "conference": it.get("conference") or team_conf.get(it.get("team"), "FBS"),
                        "_overall": it.get("overall"),
                        "_offense": it.get("offense"),
                        "_defense": it.get("defense"),
                        "_ppa_tot": it.get("total_ppa"),
                        "_ppa_off": it.get("total_offense_ppa")
                        or ((it.get("total_passing_ppa") or 0) + (it.get("total_rushing_ppa") or 0)),
                        "_ppa_def": it.get("total_defense_ppa"),
                    }
                )
        rp_df = pd.DataFrame(rp_rows).drop_duplicates(subset=["team"])
        if not rp_df.empty:
            rp_df["year"] = year
            rp_df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
            delete_rows("raw_cfbd_returning_production", "year", year)
            storage_write_dataset(rp_df, "raw_cfbd_returning_production", if_exists="append")
            rp_df = rp_df.drop(columns=["year", "retrieved_at"], errors="ignore")
    elif not rp_df.empty:
        rp_df = rp_df.drop(columns=["year", "retrieved_at"], errors="ignore")

    for _col in ["_overall", "_offense", "_defense", "_ppa_tot", "_ppa_off", "_ppa_def"]:
        if _col in rp_df.columns:
            rp_df[_col] = pd.to_numeric(rp_df[_col], errors="coerce").astype("float64")

    if not rp_df.empty:
        rp_df["wrps_offense_percent"] = rp_df["_offense"].map(_normalize_percent)
        rp_df["wrps_defense_percent"] = rp_df["_defense"].map(_normalize_percent)
        rp_df["wrps_overall_percent"] = rp_df["_overall"].map(_normalize_percent)

        if rp_df["wrps_overall_percent"].isna().all():
            rp_df["wrps_overall_percent"] = _scale_0_100(rp_df["_ppa_tot"]).round(1)
        if rp_df["wrps_offense_percent"].isna().all():
            rp_df["wrps_offense_percent"] = _scale_0_100(rp_df["_ppa_off"]).round(1)
        if rp_df["wrps_defense_percent"].isna().all():
            rp_df["wrps_defense_percent"] = _scale_0_100(rp_df["_ppa_def"]).round(1)

        rp_df["wrps_percent_0_100"] = pd.to_numeric(rp_df["wrps_overall_percent"], errors="coerce").astype("float64")
        rp_df["wrps_percent_0_100"] = rp_df["wrps_percent_0_100"].round(1)

    # Team Talent
    talent_df = read_dataset("raw_cfbd_talent")
    if not talent_df.empty:
        talent_df = talent_df.loc[talent_df.get("year") == year].copy()
    if talent_df.empty and apis.teams_api:
        key = f"talent:{year}"
        ok, data = cache.get(key)
        if ok:
            df = pd.DataFrame(data)
        else:
            try:
                items = apis.teams_api.get_talent(year=year)
                df = pd.DataFrame([{"team": x.team, "talent": float(getattr(x, "talent", 0) or 0)} for x in items or []])
                cache.set(key, df.to_dict(orient="records"))
            except Exception as e:
                print(f"[warn] talent fetch failed: {e}", file=sys.stderr)
                df = pd.DataFrame()
        if not df.empty:
            mn, mx = df["talent"].min(), df["talent"].max()
            if mx == mn:
                df["talent_score_0_100"] = 50.0
            else:
                df["talent_score_0_100"] = ((df["talent"] - mn) / (mx - mn) * 100.0).round(1)
            df["year"] = year
            df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
            delete_rows("raw_cfbd_talent", "year", year)
            storage_write_dataset(df, "raw_cfbd_talent", if_exists="append")
            talent_df = df
        else:
            talent_df = pd.DataFrame()
    if not talent_df.empty:
        talent_df = talent_df.drop(columns=["year", "retrieved_at", "talent"], errors="ignore")
        talent_df = talent_df[[c for c in ["team", "talent_score_0_100"] if c in talent_df.columns]]

    # Current season SRS ratings → rank and rank-score
    srs_cur_df = read_dataset("raw_cfbd_srs")
    if not srs_cur_df.empty:
        srs_cur_df = srs_cur_df.loc[srs_cur_df.get("year") == year].copy()
    if srs_cur_df.empty:
        try:
            if apis.ratings_api:
                srs = apis.ratings_api.get_srs(year=year)
                df = pd.DataFrame([
                    {"team": x.team, "srs_rating": getattr(x, "rating", None), "srs_rank_1_133": getattr(x, "rank", None)} for x in (srs or [])
                ])
                if not df.empty:
                    mn, mx = pd.to_numeric(df["srs_rating"], errors="coerce").min(), pd.to_numeric(df["srs_rating"], errors="coerce").max()
                    if pd.isna(mn) or pd.isna(mx) or mx == mn:
                        df["srs_score_0_100"] = 50.0
                    else:
                        df["srs_score_0_100"] = ((pd.to_numeric(df["srs_rating"], errors="coerce") - mn) / (mx - mn) * 100.0).round(1)
                    df["year"] = year
                    df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
                    delete_rows("raw_cfbd_srs", "year", year)
                    storage_write_dataset(df, "raw_cfbd_srs", if_exists="append")
                    srs_cur_df = df
        except Exception as e:
            print(f"[warn] srs fetch failed: {e}", file=sys.stderr)
            srs_cur_df = pd.DataFrame({"team": [], "srs_rating": [], "srs_rank_1_133": [], "srs_score_0_100": []})
    if not srs_cur_df.empty:
        srs_cur_df = srs_cur_df.drop(columns=["year", "retrieved_at"], errors="ignore")

    # Previous season SOS rank
    sos_df = read_dataset("raw_cfbd_sos")
    prev_season = year - 1
    if not sos_df.empty:
        sos_df = sos_df.loc[sos_df.get("season") == prev_season].copy()
    if sos_df.empty:
        try:
            if apis.ratings_api:
                sos = apis.ratings_api.get_sos(year=prev_season)
                sos_df = pd.DataFrame([{ "team": x.team, "prev_season_sos_rank_1_133": getattr(x, "rank", None)} for x in (sos or [])])
                if not sos_df.empty:
                    sos_df["season"] = prev_season
                    sos_df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
                    delete_rows("raw_cfbd_sos", "season", prev_season)
                    storage_write_dataset(sos_df, "raw_cfbd_sos", if_exists="append")
        except Exception:
            sos_df = pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})
    if 'team' not in sos_df.columns:
        sos_df = pd.DataFrame({'team': [], 'prev_season_sos_rank_1_133': []})
    if not sos_df.empty:
        sos_df = sos_df.drop(columns=["season", "retrieved_at"], errors="ignore")

    # Transfer portal net score (placeholder: unavailable via CFBD in this context)
    portal_df = pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})

    # Merge
    df = rp_df.merge(talent_df, on="team", how="outer") if not rp_df.empty else (talent_df if not talent_df.empty else pd.DataFrame())
    if df.empty and not talent_df.empty:
        df = talent_df.copy()
        df["conference"] = "FBS"
    if not df.empty:
        df = df.merge(srs_cur_df, on="team", how="left")
        df = df.merge(sos_df, on="team", how="left")
        df = df.merge(portal_df, on="team", how="left")
    else:
        seed = [
            {"team": "Kansas State", "conference": "Big 12", "wrps_percent_0_100": 60, "talent_score_0_100": 68, "portal_net_0_100": 55},
            {"team": "Iowa State", "conference": "Big 12", "wrps_percent_0_100": 55, "talent_score_0_100": 60, "portal_net_0_100": 52},
            {"team": "Hawai'i", "conference": "MWC", "wrps_percent_0_100": 48, "talent_score_0_100": 40, "portal_net_0_100": 45},
            {"team": "Stanford", "conference": "ACC", "wrps_percent_0_100": 50, "talent_score_0_100": 62, "portal_net_0_100": 43},
            {"team": "Wyoming", "conference": "MWC", "wrps_percent_0_100": 57, "talent_score_0_100": 50, "portal_net_0_100": 49},
            {"team": "Akron", "conference": "MAC", "wrps_percent_0_100": 45, "talent_score_0_100": 38, "portal_net_0_100": 42},
        ]
        df = pd.DataFrame(seed)

    if "conference" not in df.columns or df["conference"].isna().any():
        if team_conf:
            df["conference"] = df["team"].map(team_conf).fillna(df.get("conference")).fillna("FBS")
        else:
            df["conference"] = df.get("conference", "FBS")

    for col in [
        "team",
        "conference",
        "wrps_offense_percent",
        "wrps_defense_percent",
        "wrps_overall_percent",
        "wrps_percent_0_100",
        "talent_score_0_100",
        "srs_rating",
        "srs_rank_1_133",
        "srs_score_0_100",
        "prev_season_sos_rank_1_133",
        "portal_net_0_100",
        "portal_net_count",
        "portal_net_value",
    ]:
        if col not in df.columns:
            df[col] = None

    if "conference" in df.columns:
        df.sort_values(["conference", "team"], inplace=True, ignore_index=True)
    else:
        df.sort_values(["team"], inplace=True, ignore_index=True)

    return df


__all__ = ["build_team_inputs_datadriven"]
