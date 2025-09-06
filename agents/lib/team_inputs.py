from __future__ import annotations

from typing import Optional
import pandas as pd

from .cfbd_clients import CfbdClients
from .cache import ApiCache


__all__ = ["build_team_inputs_datadriven"]


def _normalize_percent(x: Optional[float]) -> Optional[float]:
    """Coerce 0..1 -> 0..100, pass-through 0..100, tolerate None/bad."""
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return x * 100.0 if x <= 1.0 else x


def _scale_0_100(series: pd.Series) -> pd.Series:
    """Min–max scale to 0..100; if constant/empty, return 50s/NaNs safely."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([None] * len(s), index=s.index)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100.0


def _df_returning_production(apis: CfbdClients, season: int) -> pd.DataFrame:
    """
    Build WRPS-like returning production for offense/defense/overall.
    Falls back to scaled PPA if explicit percent fields are missing.
    """
    # Team -> conference map (fill holes in the endpoint)
    fbs = apis.teams_api.get_fbs_teams(year=season)
    team_conf = {t.school: (t.conference or "FBS") for t in fbs}
    conferences = sorted({t.conference for t in fbs if t.conference})

    rows = []
    for conf in conferences:
        try:
            items = apis.players_api.get_returning_production(year=season, conference=conf)
        except Exception as e:
            print(f"[warn] returning production fetch failed for {conf}: {e}")
            items = []
        for it in items or []:
            rows.append(
                {
                    "team": getattr(it, "team", None),
                    "conference": getattr(it, "conference", None) or team_conf.get(getattr(it, "team", None), ""),
                    "_overall": getattr(it, "overall", None),
                    "_offense": getattr(it, "offense", None),
                    "_defense": getattr(it, "defense", None),
                    # PPA fallbacks (names vary by season/schema)
                    "_ppa_tot": getattr(it, "total_ppa", None),
                    "_ppa_off": getattr(it, "total_offense_ppa", None)
                    or (getattr(it, "total_passing_ppa", 0) + getattr(it, "total_rushing_ppa", 0)),
                    "_ppa_def": getattr(it, "total_defense_ppa", None) or getattr(it, "total_defensive_ppa", None),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])

    # Prefer explicit percents
    if not df.empty:
        df["wrps_offense_percent"] = df["_offense"].map(_normalize_percent)
        df["wrps_defense_percent"] = df["_defense"].map(_normalize_percent)
        df["wrps_overall_percent"] = df["_overall"].map(_normalize_percent)

        # If any are completely missing, fall back to scaled PPA
        if df["wrps_overall_percent"].isna().all():
            df["wrps_overall_percent"] = _scale_0_100(df["_ppa_tot"]).round(1)
        if df["wrps_offense_percent"].isna().all():
            df["wrps_offense_percent"] = _scale_0_100(df["_ppa_off"]).round(1)
        if df["wrps_defense_percent"].isna().all():
            df["wrps_defense_percent"] = _scale_0_100(df["_ppa_def"]).round(1)

        df["wrps_percent_0_100"] = pd.to_numeric(df["wrps_overall_percent"], errors="coerce").round(1)

        # Fill missing conference from fbs map
        if "conference" not in df.columns or df["conference"].isna().any():
            df["conference"] = df["team"].map(team_conf).fillna(df.get("conference")).fillna("FBS")

        keep = [
            "team",
            "conference",
            "wrps_offense_percent",
            "wrps_defense_percent",
            "wrps_overall_percent",
            "wrps_percent_0_100",
        ]
        for k in keep:
            if k not in df.columns:
                df[k] = None
        df = df[keep]

    return df


def _df_team_talent(apis: CfbdClients, season: int) -> pd.DataFrame:
    """Recruiting team-talent composite scaled to 0..100."""
    try:
        items = apis.teams_api.get_talent(year=season)
    except Exception as e:
        print(f"[warn] talent fetch failed: {e}")
        return pd.DataFrame({"team": [], "talent_score_0_100": []})

    df = pd.DataFrame([{"team": x.team, "talent": float(getattr(x, "talent", 0) or 0)} for x in items or []])
    if df.empty:
        return pd.DataFrame({"team": [], "talent_score_0_100": []})

    mn, mx = df["talent"].min(), df["talent"].max()
    if mx == mn:
        df["talent_score_0_100"] = 50.0
    else:
        df["talent_score_0_100"] = ((df["talent"] - mn) / (mx - mn) * 100.0).round(1)
    return df[["team", "talent_score_0_100"]]


def _df_prev_sos_rank(apis: CfbdClients, prior_season: int) -> pd.DataFrame:
    """Previous season opponent-strength via SRS-based average opponent rating, ranked 1..N (harder = 1)."""
    try:
        srs = apis.ratings_api.get_srs(year=prior_season)
    except Exception as e:
        print(f"[warn] srs fetch failed: {e}")
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})

    srs_map = {x.team: float(x.rating or 0) for x in srs}

    try:
        games = apis.games_api.get_games(year=prior_season, season_type="both")
    except Exception as e:
        print(f"[warn] games fetch failed: {e}")
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})

    from collections import defaultdict

    opps = defaultdict(list)
    for g in games or []:
        ht, at = getattr(g, "home_team", None), getattr(g, "away_team", None)
        if ht in srs_map and at in srs_map:
            opps[ht].append(srs_map[at])
            opps[at].append(srs_map[ht])

    rows = [{"team": t, "sos_value": sum(v) / len(v)} for t, v in opps.items() if v]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"team": [], "prev_season_sos_rank_1_133": []})

    df["prev_season_sos_rank_1_133"] = df["sos_value"].rank(ascending=False, method="min").astype(int)
    return df[["team", "prev_season_sos_rank_1_133"]]


def _df_portal_net(apis: CfbdClients, season: int) -> pd.DataFrame:
    """Crude transfer portal net metric -> count+value combined into 0..100."""
    try:
        portal = apis.players_api.get_transfer_portal(year=season)
    except Exception as e:
        print(f"[warn] portal fetch failed: {e}")
        return pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})

    from collections import defaultdict

    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    rating_in = defaultdict(float)
    rating_out = defaultdict(float)

    for p in portal or []:
        to_team = getattr(p, "destination", None) or getattr(p, "to_team", None)
        from_team = getattr(p, "origin", None) or getattr(p, "from_team", None)
        rating = getattr(p, "rating", None)
        stars = getattr(p, "stars", None)
        try:
            val = float(rating) if rating is not None else (float(stars) if stars is not None else 1.0)
        except Exception:
            val = 1.0
        if to_team:
            incoming[to_team] += 1
            rating_in[to_team] += val
        if from_team:
            outgoing[from_team] += 1
            rating_out[from_team] += val

    teams = set(list(incoming.keys()) + list(outgoing.keys()))
    rows = []
    for t in teams:
        cnt_net = incoming[t] - outgoing[t]
        val_net = rating_in[t] - rating_out[t]
        rows.append({"team": t, "portal_net_count": cnt_net, "portal_net_value": val_net})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"team": [], "portal_net_0_100": [], "portal_net_count": [], "portal_net_value": []})

    df["portal_net_0_100"] = (0.5 * _scale_0_100(df["portal_net_count"]) + 0.5 * _scale_0_100(df["portal_net_value"])).round(1)
    return df[["team", "portal_net_0_100", "portal_net_count", "portal_net_value"]]


def build_team_inputs_datadriven(season: int, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    """
    Public entrypoint used by the collector.

    Returns a dataframe with:
      team, conference,
      wrps_offense_percent, wrps_defense_percent, wrps_overall_percent, wrps_percent_0_100,
      talent_score_0_100,
      prev_season_sos_rank_1_133,
      portal_net_0_100, portal_net_count, portal_net_value
    """
    prior = season - 1

    rp = _df_returning_production(apis, season)
    talent = _df_team_talent(apis, season)
    sos = _df_prev_sos_rank(apis, prior)
    portal = _df_portal_net(apis, season)

    # Merge on team
    df = (
        rp.merge(talent, on="team", how="left")
        .merge(sos, on="team", how="left")
        .merge(portal, on="team", how="left")
    )

    # Sort and final tidy
    if "conference" in df.columns:
        df.sort_values(["conference", "team"], inplace=True, ignore_index=True)
    else:
        df.sort_values(["team"], inplace=True, ignore_index=True)

    # Ensure expected columns exist (in case of sparse upstream)
    expected = [
        "team",
        "conference",
        "wrps_offense_percent",
        "wrps_defense_percent",
        "wrps_overall_percent",
        "wrps_percent_0_100",
        "talent_score_0_100",
        "prev_season_sos_rank_1_133",
        "portal_net_0_100",
        "portal_net_count",
        "portal_net_value",
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = None

    return df[expected].copy()