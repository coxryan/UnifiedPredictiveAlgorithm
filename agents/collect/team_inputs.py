from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .cfbd_clients import CfbdClients
from .cache import ApiCache
from .helpers import _normalize_percent, _scale_0_100
from .stats_cfbd import build_team_stat_features
from .config import _dbg
from agents.storage import read_dataset, write_dataset as storage_write_dataset, delete_rows


_PERCENTILE_THRESHOLDS: List[Tuple[float, str]] = [
    (99.0, "A+"),
    (95.0, "A"),
    (90.0, "A-"),
    (80.0, "B+"),
    (70.0, "B"),
    (60.0, "B-"),
    (50.0, "C+"),
    (40.0, "C"),
    (30.0, "C-"),
    (20.0, "D+"),
    (10.0, "D"),
    (0.0, "D-"),
]


def _letter_from_percentile(value: Any) -> str:
    try:
        pct = float(value)
    except Exception:
        return ""
    if pd.isna(pct):
        return ""
    for threshold, grade in _PERCENTILE_THRESHOLDS:
        if pct >= threshold:
            return grade
    return "F"


_AVAIL_OFF_POS = {
    "QB",
    "RB",
    "TB",
    "HB",
    "FB",
    "WR",
    "TE",
    "H",
    "Y",
    "X",
    "Z",
    "OL",
    "LT",
    "LG",
    "C",
    "RG",
    "RT",
}

_AVAIL_DEF_POS = {
    "DL",
    "DE",
    "DT",
    "NT",
    "EDGE",
    "LB",
    "MLB",
    "OLB",
    "WLB",
    "SLB",
    "CB",
    "DB",
    "FS",
    "SS",
    "NICKEL",
    "STAR",
}

_AVAIL_ST_POS = {
    "K",
    "P",
    "PK",
    "LS",
    "KR",
    "PR",
    "ST",
}


def _load_cfbd_player_usage(
    year: int,
    apis: CfbdClients,
    cache: ApiCache,
    team_conf: Dict[str, str],
) -> pd.DataFrame:
    usage_df = read_dataset("raw_cfbd_player_usage")
    if not usage_df.empty:
        usage_df = usage_df.loc[usage_df.get("year") == year].copy()
    if not usage_df.empty:
        usage_df = usage_df.drop(columns=["year", "retrieved_at"], errors="ignore")
        _dbg(f"team_inputs: using cached player usage rows={len(usage_df)} year={year}")
        return usage_df

    if not apis.players_api or not team_conf:
        _dbg(f"team_inputs: player usage unavailable (players_api={bool(apis.players_api)} team_conf={len(team_conf)}) year={year}")
        return pd.DataFrame()

    conferences = sorted({c for c in team_conf.values() if c})
    rows: List[Dict[str, Any]] = []
    for conf in conferences:
        cache_key = f"cfbd:player_usage:{year}:{conf}"
        ok, cached = cache.get(cache_key)
        if ok and cached:
            rows.extend(cached)
            continue
        try:
            items = apis.players_api.get_player_usage(year=year, conference=conf)
        except Exception as exc:  # pragma: no cover - network failure path
            print(f"[warn] player usage fetch failed for {conf}: {exc}", file=sys.stderr)
            items = []
        serialised: List[Dict[str, Any]] = []
        for it in items or []:
            usage = getattr(it, "usage", None)
            serialised.append(
                {
                    "team": getattr(it, "team", None),
                    "conference": getattr(it, "conference", None) or team_conf.get(getattr(it, "team", None), "FBS"),
                    "player_id": getattr(it, "id", None),
                    "player_name": getattr(it, "name", None),
                    "position": (getattr(it, "position", None) or "").upper(),
                    "usage_overall": getattr(usage, "overall", None) if usage is not None else None,
                    "usage_rush": getattr(usage, "rush", None) if usage is not None else None,
                    "usage_pass": getattr(usage, "var_pass", None) if usage is not None else None,
                    "usage_first_down": getattr(usage, "first_down", None) if usage is not None else None,
                    "usage_standard_downs": getattr(usage, "standard_downs", None) if usage is not None else None,
                    "usage_passing_downs": getattr(usage, "passing_downs", None) if usage is not None else None,
                }
            )
        if serialised:
            cache.set(cache_key, serialised)
            rows.extend(serialised)
    if not rows:
        return pd.DataFrame()
    usage_df = pd.DataFrame(rows)
    usage_df["year"] = year
    usage_df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
    delete_rows("raw_cfbd_player_usage", "year", year)
    storage_write_dataset(usage_df, "raw_cfbd_player_usage", if_exists="append")
    usage_df = usage_df.drop(columns=["year", "retrieved_at"], errors="ignore")
    _dbg(f"team_inputs: fetched player usage rows={len(usage_df)} conferences={len(conferences)} year={year}")
    return usage_df


def _build_availability_features(
    usage_df: pd.DataFrame,
    team_conf: Dict[str, str],
    default_score: float = 0.65,
) -> pd.DataFrame:
    if usage_df is None or usage_df.empty:
        if not team_conf:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "team": team,
                    "availability_source": "default",
                    "availability_offense_score": round(default_score * 100, 1),
                    "availability_defense_score": round(default_score * 100, 1),
                    "availability_special_score": round(default_score * 100, 1),
                    "availability_qb_score": round(default_score * 100, 1),
                    "availability_overall_score": round(default_score * 100, 1),
                    "availability_flag_qb_low": 0,
                    "availability_off_depth": 0,
                    "availability_def_depth": 0,
                }
                for team in sorted(team_conf.keys())
            ]
        )

    df = usage_df.copy()
    df["team"] = df["team"].astype(str)
    for col in [
        "usage_overall",
        "usage_rush",
        "usage_pass",
        "usage_first_down",
        "usage_standard_downs",
        "usage_passing_downs",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    records: List[Dict[str, Any]] = []
    grouped = df.groupby("team")

    def _score_for_positions(grp: pd.DataFrame, positions: set[str], top_n: int, fallback: float) -> tuple[float, int]:
        subset = grp.loc[grp["position"].isin(positions)]
        if subset.empty:
            return fallback, 0
        values = subset["usage_overall"].sort_values(ascending=False)
        if values.empty:
            return fallback, 0
        top_vals = values.head(top_n)
        mean_val = float(top_vals.mean()) if not top_vals.empty else fallback
        contributor_count = int((values >= 0.40).sum())
        return mean_val, contributor_count

    for team, grp in grouped:
        grp = grp.copy()
        grp["position"] = grp["position"].astype(str).str.upper()
        off_score_raw, off_depth = _score_for_positions(grp, _AVAIL_OFF_POS, 5, default_score)
        def_score_raw, def_depth = _score_for_positions(grp, _AVAIL_DEF_POS, 5, default_score)
        st_score_raw, _ = _score_for_positions(grp, _AVAIL_ST_POS, 3, default_score)
        qb_score_raw, _ = _score_for_positions(grp, {"QB"}, 1, default_score)
        overall_score = 0.6 * off_score_raw + 0.3 * def_score_raw + 0.1 * st_score_raw

        def _sum_top(series: pd.Series, n: int) -> float:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return 0.0
            return float(numeric.sort_values(ascending=False).head(n).sum())

        off_usage = grp.loc[grp["position"].isin(_AVAIL_OFF_POS), "usage_overall"]
        def_usage = grp.loc[grp["position"].isin(_AVAIL_DEF_POS), "usage_overall"]
        qb_usage_overall = grp.loc[grp["position"] == "QB", "usage_overall"]
        qb_usage_pass = grp.loc[grp["position"] == "QB", "usage_pass"] if "usage_pass" in grp.columns else pd.Series([], dtype=float)

        top3_off_usage = _sum_top(off_usage, 3)
        top3_def_usage = _sum_top(def_usage, 3)
        qb_usage_val = float(pd.to_numeric(qb_usage_overall, errors="coerce").max()) if not qb_usage_overall.empty else 0.0
        qb_pass_usage_val = float(pd.to_numeric(qb_usage_pass, errors="coerce").max()) if not qb_usage_pass.empty else 0.0

        record = {
            "team": team,
            "availability_source": "cfbd_usage",
            "availability_offense_score": round(max(0.0, min(off_score_raw, 1.0)) * 100, 1),
            "availability_defense_score": round(max(0.0, min(def_score_raw, 1.0)) * 100, 1),
            "availability_special_score": round(max(0.0, min(st_score_raw, 1.0)) * 100, 1),
            "availability_qb_score": round(max(0.0, min(qb_score_raw, 1.0)) * 100, 1),
            "availability_overall_score": round(max(0.0, min(overall_score, 1.0)) * 100, 1),
            "availability_flag_qb_low": int(qb_score_raw < 0.40),
            "availability_off_depth": off_depth,
            "availability_def_depth": def_depth,
            "availability_off_top3_usage_pct": round(top3_off_usage * 100, 1),
            "availability_def_top3_usage_pct": round(top3_def_usage * 100, 1),
            "availability_qb_usage_pct": round(qb_usage_val * 100, 1),
            "availability_qb_pass_usage_pct": round(qb_pass_usage_val * 100, 1),
        }
        records.append(record)

    if not team_conf:
        return pd.DataFrame(records)

    known_teams = {rec["team"] for rec in records}
    for team in sorted(team_conf.keys()):
        if team in known_teams:
            continue
        records.append(
            {
                "team": team,
                "availability_source": "default",
                "availability_offense_score": round(default_score * 100, 1),
                "availability_defense_score": round(default_score * 100, 1),
                "availability_special_score": round(default_score * 100, 1),
                "availability_qb_score": round(default_score * 100, 1),
                "availability_overall_score": round(default_score * 100, 1),
                "availability_flag_qb_low": 0,
                "availability_off_depth": 0,
                "availability_def_depth": 0,
                "availability_off_top3_usage_pct": round(default_score * 100, 1),
                "availability_def_top3_usage_pct": round(default_score * 100, 1),
                "availability_qb_usage_pct": round(default_score * 100, 1),
                "availability_qb_pass_usage_pct": round(default_score * 100, 1),
            }
        )

    return pd.DataFrame(records)


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

    _dbg(f"team_inputs: team_conf entries={len(team_conf)} year={year}")

    usage_df = _load_cfbd_player_usage(year, apis, cache, team_conf)
    availability_df = _build_availability_features(usage_df, team_conf)
    _dbg(f"team_inputs: availability rows={len(availability_df)} year={year}")

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

    # SP+ ratings (opponent-adjusted efficiency, includes SOS and rankings)
    sp_df = read_dataset("raw_cfbd_sp")
    if not sp_df.empty:
        sp_df = sp_df.loc[sp_df.get("year") == year].copy()
    if sp_df.empty:
        try:
            if apis.ratings_api:
                sp_payload = apis.ratings_api.get_sp(year=year)
                sp_rows: List[Dict[str, Any]] = []
                for item in sp_payload or []:
                    if not getattr(item, "team", None):
                        continue
                    offense = getattr(item, "offense", None)
                    defense = getattr(item, "defense", None)
                    sp_rows.append(
                        {
                            "team": getattr(item, "team", None),
                            "conference": getattr(item, "conference", None),
                            "sp_rating": getattr(item, "rating", None),
                            "sp_ranking": getattr(item, "ranking", None),
                            "sp_sos": getattr(item, "sos", None),
                            "sp_off_rating": getattr(offense, "rating", None) if offense is not None else None,
                            "sp_off_success": getattr(offense, "success", None) if offense is not None else None,
                            "sp_def_rating": getattr(defense, "rating", None) if defense is not None else None,
                            "sp_def_success": getattr(defense, "success", None) if defense is not None else None,
                        }
                    )
                sp_df = pd.DataFrame(sp_rows)
                if not sp_df.empty:
                    numeric_cols = [
                        "sp_rating",
                        "sp_ranking",
                        "sp_sos",
                        "sp_off_rating",
                        "sp_off_success",
                        "sp_def_rating",
                        "sp_def_success",
                    ]
                    for col in numeric_cols:
                        if col in sp_df.columns:
                            sp_df[col] = pd.to_numeric(sp_df[col], errors="coerce")
                    sp_df["year"] = year
                    sp_df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
                    delete_rows("raw_cfbd_sp", "year", year)
                    storage_write_dataset(sp_df, "raw_cfbd_sp", if_exists="append")
        except Exception as exc:
            print(f"[warn] SP+ fetch failed: {exc}", file=sys.stderr)
            sp_df = pd.DataFrame()
    if not sp_df.empty:
        sp_df = sp_df.drop(columns=["year", "retrieved_at"], errors="ignore")
        for col in ["sp_rating", "sp_sos", "sp_off_rating", "sp_off_success", "sp_def_rating", "sp_def_success"]:
            if col in sp_df.columns:
                sp_df[col] = pd.to_numeric(sp_df[col], errors="coerce")
        # Convert SP ratings to 0-100 scales for blending
        if "sp_rating" in sp_df.columns:
            sp_df["sp_rating_0_100"] = _scale_0_100(sp_df["sp_rating"]).round(1)
        if "sp_sos" in sp_df.columns:
            sp_df["sp_sos_0_100"] = _scale_0_100(sp_df["sp_sos"]).round(1)
        if "sp_off_rating" in sp_df.columns:
            sp_df["sp_off_rating_0_100"] = _scale_0_100(sp_df["sp_off_rating"]).round(1)
        if "sp_def_rating" in sp_df.columns:
            # Defensive SP ratings are typically negative when better. Multiply by -1 before scaling.
            sp_df["sp_def_rating_0_100"] = _scale_0_100(sp_df["sp_def_rating"] * -1.0).round(1)
        if "sp_off_success" in sp_df.columns:
            sp_df["sp_off_success_0_100"] = _scale_0_100(sp_df["sp_off_success"]).round(1)
        if "sp_def_success" in sp_df.columns:
            sp_df["sp_def_success_0_100"] = _scale_0_100(sp_df["sp_def_success"] * -1.0).round(1)
    _dbg(f"team_inputs: sp_df rows={len(sp_df)} year={year}")

    # FPI ratings
    fpi_df = read_dataset("raw_cfbd_fpi")
    if not fpi_df.empty:
        fpi_df = fpi_df.loc[fpi_df.get("year") == year].copy()
    if fpi_df.empty and apis.ratings_api:
        try:
            fpi_payload = apis.ratings_api.get_fpi(year=year)
            fpi_rows = []
            for item in fpi_payload or []:
                fpi_rows.append(
                    {
                        "team": getattr(item, "team", None),
                        "conference": getattr(item, "conference", None),
                        "fpi_rating": getattr(item, "fpi", None),
                        "fpi_game_control_rank": getattr(getattr(item, "resume_ranks", None), "game_control", None),
                        "fpi_sos_rank": getattr(getattr(item, "resume_ranks", None), "strength_of_schedule", None),
                        "fpi_remaining_sos_rank": getattr(getattr(item, "resume_ranks", None), "remaining_strength_of_schedule", None),
                        "fpi_off_eff": getattr(getattr(item, "efficiencies", None), "offense", None),
                        "fpi_def_eff": getattr(getattr(item, "efficiencies", None), "defense", None),
                        "fpi_st_eff": getattr(getattr(item, "efficiencies", None), "special_teams", None),
                        "fpi_overall_eff": getattr(getattr(item, "efficiencies", None), "overall", None),
                    }
                )
            fpi_df = pd.DataFrame(fpi_rows)
            if not fpi_df.empty:
                fpi_df["year"] = year
                fpi_df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
                delete_rows("raw_cfbd_fpi", "year", year)
                storage_write_dataset(fpi_df, "raw_cfbd_fpi", if_exists="append")
        except Exception as exc:
            print(f"[warn] FPI fetch failed: {exc}", file=sys.stderr)
            fpi_df = pd.DataFrame()
    if not fpi_df.empty:
        fpi_df = fpi_df.drop(columns=["year", "retrieved_at"], errors="ignore")
        for col in [
            "fpi_rating",
            "fpi_game_control_rank",
            "fpi_sos_rank",
            "fpi_remaining_sos_rank",
            "fpi_off_eff",
            "fpi_def_eff",
            "fpi_st_eff",
            "fpi_overall_eff",
        ]:
            if col in fpi_df.columns:
                fpi_df[col] = pd.to_numeric(fpi_df[col], errors="coerce")
        if "fpi_rating" in fpi_df.columns:
            fpi_df["fpi_rating_0_100"] = _scale_0_100(fpi_df["fpi_rating"]).round(1)
        if "fpi_overall_eff" in fpi_df.columns:
            fpi_df["fpi_overall_eff_0_100"] = _scale_0_100(fpi_df["fpi_overall_eff"]).round(1)
        if "fpi_off_eff" in fpi_df.columns:
            fpi_df["fpi_off_eff_0_100"] = _scale_0_100(fpi_df["fpi_off_eff"]).round(1)
        if "fpi_def_eff" in fpi_df.columns:
            fpi_df["fpi_def_eff_0_100"] = _scale_0_100(fpi_df["fpi_def_eff"] * -1.0).round(1)
        if "fpi_st_eff" in fpi_df.columns:
            fpi_df["fpi_st_eff_0_100"] = _scale_0_100(fpi_df["fpi_st_eff"]).round(1)
    _dbg(f"team_inputs: fpi rows={len(fpi_df)} year={year}")

    # Elo ratings
    elo_df = read_dataset("raw_cfbd_elo")
    if not elo_df.empty:
        elo_df = elo_df.loc[elo_df.get("year") == year].copy()
    if elo_df.empty and apis.ratings_api:
        try:
            elo_payload = apis.ratings_api.get_elo(year=year)
            elo_rows = [
                {
                    "team": getattr(item, "team", None),
                    "conference": getattr(item, "conference", None),
                    "elo_rating": getattr(item, "elo", None),
                }
                for item in (elo_payload or [])
            ]
            elo_df = pd.DataFrame(elo_rows)
            if not elo_df.empty:
                elo_df["year"] = year
                elo_df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
                delete_rows("raw_cfbd_elo", "year", year)
                storage_write_dataset(elo_df, "raw_cfbd_elo", if_exists="append")
        except Exception as exc:
            print(f"[warn] Elo fetch failed: {exc}", file=sys.stderr)
            elo_df = pd.DataFrame()
    if not elo_df.empty:
        elo_df = elo_df.drop(columns=["year", "retrieved_at"], errors="ignore")
        if "elo_rating" in elo_df.columns:
            elo_df["elo_rating"] = pd.to_numeric(elo_df["elo_rating"], errors="coerce")
            elo_df["elo_rating_0_100"] = _scale_0_100(elo_df["elo_rating"]).round(1)
    _dbg(f"team_inputs: elo rows={len(elo_df)} year={year}")

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
    _dbg(f"team_inputs: sos rows={len(sos_df)} prev_season={prev_season}")

    # Statistical feature library (offense/defense/special teams efficiency)
    stats_df = build_team_stat_features(year, apis, cache)
    _dbg(f"team_inputs: stats_df rows={len(stats_df)} year={year}")

    # Transfer portal net score (placeholder: unavailable via CFBD in this context)
    portal_df = pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})

    # Merge
    df = rp_df.merge(talent_df, on="team", how="outer") if not rp_df.empty else (talent_df if not talent_df.empty else pd.DataFrame())
    if df.empty and not talent_df.empty:
        df = talent_df.copy()
        df["conference"] = "FBS"
    if not df.empty:
        df = df.merge(srs_cur_df, on="team", how="left")
        if not sp_df.empty:
            df = df.merge(sp_df, on="team", how="left")
        if not fpi_df.empty:
            df = df.merge(fpi_df, on="team", how="left")
        if not elo_df.empty:
            df = df.merge(elo_df, on="team", how="left")
        df = df.merge(sos_df, on="team", how="left")
        if not stats_df.empty:
            df = df.merge(stats_df, on="team", how="left")
        if not availability_df.empty:
            df = df.merge(availability_df, on="team", how="left")
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

    _dbg(f"team_inputs: merged base frame rows={len(df)} year={year}")

    if team_conf:
        mapped_conf = df["team"].map(team_conf)
        if "conference" in df.columns:
            df["conference"] = mapped_conf.fillna(df["conference"])
        else:
            df["conference"] = mapped_conf
    if "conference" not in df.columns:
        df["conference"] = "FBS"
    df["conference"] = df["conference"].fillna("FBS")
    conf_missing = int(df["conference"].isna().sum())
    if conf_missing:
        _dbg(f"team_inputs: conference still missing rows={conf_missing} year={year}")

    if not df.empty:
        def _col(name: str, default: float = 50.0) -> pd.Series:
            if name in df.columns:
                series = pd.to_numeric(df[name], errors="coerce")
            else:
                series = pd.Series(default, index=df.index, dtype="float64")
            return series.fillna(default)

        off_idx = _col("stat_off_index_0_100")
        def_idx = _col("stat_def_index_0_100")
        st_idx = _col("stat_st_index_0_100")
        wrps_off = _col("wrps_offense_percent")
        wrps_def = _col("wrps_defense_percent")
        talent = _col("talent_score_0_100")
        srs = _col("srs_score_0_100")

        off_success = _col("stat_off_success")
        off_expl = _col("stat_off_explosiveness")
        def_success = _col("stat_def_success")
        def_expl = _col("stat_def_explosiveness")
        off_ppd_adj = _col("stat_off_ppd_adj")
        def_ppd_adj = _col("stat_def_ppd_adj")
        off_success_adj = _col("stat_off_success_adj")
        def_success_adj = _col("stat_def_success_adj")
        off_ppa_adj = _col("stat_off_ppa_adj")
        def_ppa_adj = _col("stat_def_ppa_adj")
        off_ypp_adj = _col("stat_off_ypp_adj")
        def_ypp_adj = _col("stat_def_ypp_adj")
        sp_off_rating = _col("sp_off_rating_0_100")
        sp_def_rating = _col("sp_def_rating_0_100")

        def _blend(series: pd.Series) -> pd.Series:
            return series.clip(lower=0.0, upper=100.0)

        df["grade_qb_score"] = _blend(
            0.25 * off_idx
            + 0.20 * off_ppd_adj
            + 0.15 * off_success_adj
            + 0.15 * wrps_off
            + 0.15 * talent
            + 0.10 * srs
        )
        df["grade_wr_score"] = _blend(
            0.30 * off_idx
            + 0.25 * off_expl
            + 0.15 * off_success_adj
            + 0.15 * wrps_off
            + 0.15 * sp_off_rating
        )
        df["grade_rb_score"] = _blend(
            0.25 * off_idx
            + 0.25 * off_success
            + 0.20 * off_ppd_adj
            + 0.15 * wrps_off
            + 0.15 * sp_off_rating
        )
        df["grade_ol_score"] = _blend(
            0.25 * off_idx
            + 0.25 * off_success
            + 0.20 * off_ppd_adj
            + 0.15 * talent
            + 0.15 * srs
        )
        df["grade_dl_score"] = _blend(
            0.25 * def_idx
            + 0.25 * def_ppd_adj
            + 0.20 * def_expl
            + 0.15 * wrps_def
            + 0.15 * sp_def_rating
        )
        df["grade_lb_score"] = _blend(
            0.25 * def_idx
            + 0.25 * def_success
            + 0.20 * def_ppd_adj
            + 0.15 * wrps_def
            + 0.15 * sp_def_rating
        )
        df["grade_db_score"] = _blend(
            0.25 * def_idx
            + 0.30 * def_success_adj
            + 0.20 * def_expl
            + 0.15 * wrps_def
            + 0.10 * sp_def_rating
        )
        df["grade_st_score"] = _blend(0.60 * st_idx + 0.20 * talent + 0.20 * wrps_off)

        grade_keys = ["qb", "wr", "rb", "ol", "dl", "lb", "db", "st"]
        for key in grade_keys:
            score_col = f"grade_{key}_score"
            pct_col = f"grade_{key}_percentile"
            letter_col = f"grade_{key}_letter"
            pct = df[score_col].rank(pct=True, ascending=True) * 100.0
            df[pct_col] = pct.round(1)
            df[letter_col] = df[pct_col].map(_letter_from_percentile)

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
        "stat_off_ppg",
        "stat_off_ypp",
        "stat_off_success",
        "stat_off_explosiveness",
        "stat_def_ppg",
        "stat_def_ypp",
        "stat_def_success",
        "stat_def_explosiveness",
        "stat_st_points_per_play",
        "stat_off_index_0_100",
        "stat_def_index_0_100",
        "stat_st_index_0_100",
        "stat_off_success_adj",
        "stat_def_success_adj",
        "stat_off_ppd_adj",
        "stat_def_ppd_adj",
        "stat_off_ppa_adj",
        "stat_def_ppa_adj",
        "stat_off_ypp_adj",
        "stat_def_ypp_adj",
        "stat_off_pd_success",
        "stat_off_sd_success",
        "stat_off_points_per_opp",
        "stat_off_total_opps",
        "stat_off_havoc_front",
        "stat_off_havoc_db",
        "stat_off_field_pos_avg_start",
        "stat_off_field_pos_pred_pts",
        "stat_def_pd_success",
        "stat_def_sd_success",
        "stat_def_havoc_front",
        "stat_def_havoc_db",
        "stat_def_field_pos_avg_start",
        "stat_def_field_pos_pred_pts",
        "advanced_off_success_rate",
        "advanced_off_ppd",
        "advanced_off_ppa",
        "advanced_off_ypp",
        "advanced_off_pd_success",
        "advanced_off_sd_success",
        "advanced_off_points_per_opp",
        "advanced_off_total_opps",
        "advanced_off_havoc_front",
        "advanced_off_havoc_db",
        "advanced_off_field_pos_avg_start",
        "advanced_off_field_pos_pred_pts",
        "advanced_def_success_rate",
        "advanced_def_ppd",
        "advanced_def_ppa",
        "advanced_def_ypp",
        "advanced_def_pd_success",
        "advanced_def_sd_success",
        "advanced_def_havoc_front",
        "advanced_def_havoc_db",
        "advanced_def_field_pos_avg_start",
        "advanced_def_field_pos_pred_pts",
        "rolling_off_success_rate_4",
        "rolling_off_ppd_4",
        "rolling_off_ppa_4",
        "rolling_off_ypp_4",
        "rolling_def_success_rate_4",
        "rolling_def_ppd_4",
        "rolling_def_ppa_4",
        "rolling_def_ypp_4",
        "availability_source",
        "availability_offense_score",
        "availability_defense_score",
        "availability_special_score",
        "availability_qb_score",
        "availability_overall_score",
        "availability_flag_qb_low",
        "availability_off_depth",
        "availability_def_depth",
        "availability_off_top3_usage_pct",
        "availability_def_top3_usage_pct",
        "availability_qb_usage_pct",
        "availability_qb_pass_usage_pct",
        "grade_qb_score",
        "grade_qb_percentile",
        "grade_qb_letter",
        "grade_wr_score",
        "grade_wr_percentile",
        "grade_wr_letter",
        "grade_rb_score",
        "grade_rb_percentile",
        "grade_rb_letter",
        "grade_ol_score",
        "grade_ol_percentile",
        "grade_ol_letter",
        "grade_dl_score",
        "grade_dl_percentile",
        "grade_dl_letter",
        "grade_lb_score",
        "grade_lb_percentile",
        "grade_lb_letter",
        "grade_db_score",
        "grade_db_percentile",
        "grade_db_letter",
        "grade_st_score",
        "grade_st_percentile",
        "grade_st_letter",
        "portal_net_0_100",
        "portal_net_count",
        "portal_net_value",
        "sp_rating",
        "sp_rating_0_100",
        "sp_ranking",
        "sp_sos",
        "sp_sos_0_100",
        "sp_off_rating",
        "sp_off_rating_0_100",
        "sp_off_success",
        "sp_off_success_0_100",
        "sp_def_rating",
        "sp_def_rating_0_100",
        "sp_def_success",
        "sp_def_success_0_100",
        "fpi_rating",
        "fpi_rating_0_100",
        "fpi_game_control_rank",
        "fpi_sos_rank",
        "fpi_remaining_sos_rank",
        "fpi_off_eff",
        "fpi_off_eff_0_100",
        "fpi_def_eff",
        "fpi_def_eff_0_100",
        "fpi_st_eff",
        "fpi_st_eff_0_100",
        "fpi_overall_eff",
        "fpi_overall_eff_0_100",
        "elo_rating",
        "elo_rating_0_100",
    ]:
        if col not in df.columns:
            df[col] = None

    if "conference" in df.columns:
        df.sort_values(["conference", "team"], inplace=True, ignore_index=True)
    else:
        df.sort_values(["team"], inplace=True, ignore_index=True)

    return df


__all__ = ["build_team_inputs_datadriven"]
