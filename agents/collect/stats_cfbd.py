from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from .cache import ApiCache
from .cfbd_clients import CfbdClients
from .config import _dbg
from agents.storage import read_dataset, write_dataset as storage_write_dataset, delete_rows

_STAT_FEATURE_MAP: Dict[str, str] = {
    "offense.points_per_game": "stat_off_ppg",
    "offense.pointspergame": "stat_off_ppg",
    "offense.points": "stat_off_ppg",
    "offense.yards_per_play": "stat_off_ypp",
    "offense.yardsperplay": "stat_off_ypp",
    "offense.ypp": "stat_off_ypp",
    "offense.success_rate": "stat_off_success",
    "offense.successrate": "stat_off_success",
    "offense.explosiveness": "stat_off_explosiveness",
    "offense.explosive_rate": "stat_off_explosiveness",
    "defense.points_per_game": "stat_def_ppg",
    "defense.pointspergame": "stat_def_ppg",
    "defense.yards_per_play": "stat_def_ypp",
    "defense.yardsperplay": "stat_def_ypp",
    "defense.success_rate": "stat_def_success",
    "defense.successrate": "stat_def_success",
    "defense.explosiveness": "stat_def_explosiveness",
    "defense.explosive_rate": "stat_def_explosiveness",
    "specialteams.points_per_play": "stat_st_points_per_play",
    "specialteams.pointsperplay": "stat_st_points_per_play",
    "special_teams.points_per_play": "stat_st_points_per_play",
}

_INVERT_METRICS: frozenset[str] = frozenset(
    {
        "stat_def_ppg",
        "stat_def_ypp",
        "stat_def_success",
        "stat_def_explosiveness",
    }
)

_ADVANCED_DATASET = "raw_cfbd_team_stats_advanced"

_EXPECTED_POINTS_PER_YARD = 0.07  # Rough heuristic; 1 expected point ≈ 14 yards, i.e. 0.07 pts/yard.


def _fetch_team_stats(year: int, apis: CfbdClients, cache: ApiCache) -> List[Dict[str, Any]]:
    cache_key = f"team-stats:{year}"
    ok, cached = cache.get(cache_key)
    if ok and isinstance(cached, list):
        return cached
    if not getattr(apis, "stats_api", None):
        return []
    try:
        payload = apis.stats_api.get_team_stats(year=year)
    except Exception as exc:  # pragma: no cover
        _dbg(f"team stats fetch failed year={year} err={exc}")
        return []
    rows: List[Dict[str, Any]] = []
    for item in payload or []:
        stat_name = getattr(item, "stat_name", None)
        raw_value = getattr(item, "stat_value", None)
        try:
            stat_value = float(raw_value)
        except (TypeError, ValueError):
            stat_value = None
        rows.append(
            {
                "team": getattr(item, "team", None),
                "conference": getattr(item, "conference", None),
                "stat_name": stat_name,
                "stat_value": stat_value,
            }
        )
    cache.set(cache_key, rows)
    return rows


def _normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c.startswith("stat_")]
    if not feature_cols:
        return df
    out = df.copy()
    for col in feature_cols:
        series = pd.to_numeric(out[col], errors="coerce")
        finite = series.replace([pd.NA, pd.NaT], pd.NA).dropna().astype("float64")
        if finite.empty:
            out[col] = pd.NA
            continue
        mn, mx = float(finite.min()), float(finite.max())
        if mx == mn:
            scaled = pd.Series(50.0, index=series.index, dtype="float64")
        else:
            scaled = (series - mn) / (mx - mn) * 100.0
        if col in _INVERT_METRICS:
            scaled = 100.0 - scaled
        out[col] = scaled.round(2)
    return out


def _prepare_feature_frame(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df is None or stats_df.empty:
        return pd.DataFrame(columns=["team"] + list(_STAT_FEATURE_MAP.values()))
    merged = pd.DataFrame()
    for raw_name, feature in _STAT_FEATURE_MAP.items():
        mask = stats_df["stat_name"].str.replace(" ", "_").str.lower() == raw_name.replace(" ", "_").lower()
        if not mask.any():
            continue
        subset = stats_df.loc[mask, ["team", "stat_value"]].copy()
        subset = subset.rename(columns={"stat_value": feature})
        merged = subset if merged.empty else merged.merge(subset, on="team", how="outer")
    if merged.empty:
        cols = ["team"] + list(_STAT_FEATURE_MAP.values())
        return pd.DataFrame(columns=cols)
    merged = _normalize_feature_columns(merged)
    merged.sort_values("team", inplace=True, ignore_index=True)
    return merged


def build_team_stat_features(year: int, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    raw = read_dataset("raw_cfbd_team_stats")
    stats_df = pd.DataFrame()
    if not raw.empty:
        raw = raw.loc[raw.get("season") == year].copy()
        if not raw.empty:
            stats_df = raw.drop(columns=["season", "retrieved_at"], errors="ignore")
    if stats_df.empty:
        rows = _fetch_team_stats(year, apis, cache)
        if rows:
            stats_df = pd.DataFrame(rows)
            store_df = stats_df.copy()
            store_df["season"] = year
            store_df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
            delete_rows("raw_cfbd_team_stats", "season", year)
            storage_write_dataset(store_df, "raw_cfbd_team_stats", if_exists="append")
        else:
            stats_df = pd.DataFrame(columns=["team", "category", "stat_name", "stat_value"])
    features = _prepare_feature_frame(stats_df)
    if not features.empty:
        store_feats = features.copy()
        store_feats["season"] = year
        store_feats["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
        delete_rows("raw_cfbd_team_stats_features", "season", year)
        storage_write_dataset(store_feats, "raw_cfbd_team_stats_features", if_exists="append")
        features = features.drop(columns=[c for c in ["season", "retrieved_at"] if c in features.columns], errors="ignore")
    else:
        cached_feats = read_dataset("raw_cfbd_team_stats_features")
        if not cached_feats.empty:
            cached_feats = cached_feats.loc[cached_feats.get("season") == year].copy()
            if not cached_feats.empty:
                features = cached_feats.drop(columns=["season", "retrieved_at"], errors="ignore")

    advanced = _build_advanced_metrics(year, apis, cache)
    if features.empty:
        features = advanced
    elif not advanced.empty:
        features = features.merge(advanced, on="team", how="outer")
    return features


def _safe_div(num: Any, denom: Any) -> Optional[float]:
    try:
        n = float(num)
        d = float(denom)
        if d == 0.0:
            return None
        return n / d
    except (TypeError, ValueError):
        return None


def _scale_series(series: pd.Series, invert: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series([pd.NA] * len(series))
    mn, mx = float(finite.min()), float(finite.max())
    if mx == mn:
        scaled = pd.Series(50.0, index=numeric.index, dtype="float64")
    else:
        scaled = (numeric - mn) / (mx - mn) * 100.0
    if invert:
        scaled = 100.0 - scaled
    return scaled.round(2)


def _build_advanced_metrics(year: int, apis: CfbdClients, cache: ApiCache) -> pd.DataFrame:
    """Fetch opponent-adjusted efficiency metrics (points per drive, success, PPA)."""
    cached = read_dataset(_ADVANCED_DATASET)
    advanced_df = pd.DataFrame()
    if not cached.empty:
        cached_year = cached.loc[cached.get("season") == year].copy()
        if not cached_year.empty:
            advanced_df = cached_year.drop(columns=["season", "retrieved_at"], errors="ignore")
    if advanced_df.empty and getattr(apis, "stats_api", None):
        try:
            payload = apis.stats_api.get_advanced_season_stats(year=year)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - network failure handled upstream
            payload = []
        rows: List[Dict[str, Any]] = []
        for item in payload or []:
            team = getattr(item, "team", None)
            if not team:
                continue
            offense = getattr(item, "offense", None)
            defense = getattr(item, "defense", None)
            off_success = getattr(offense, "success_rate", None) if offense is not None else None
            off_total_ppa = getattr(offense, "total_ppa", None) if offense is not None else None
            off_drives = getattr(offense, "drives", None) if offense is not None else None
            off_ppa = getattr(offense, "ppa", None) if offense is not None else None
            off_plays = getattr(offense, "plays", None) if offense is not None else None

            def_success = getattr(defense, "success_rate", None) if defense is not None else None
            def_total_ppa = getattr(defense, "total_ppa", None) if defense is not None else None
            def_drives = getattr(defense, "drives", None) if defense is not None else None
            def_ppa = getattr(defense, "ppa", None) if defense is not None else None
            def_plays = getattr(defense, "plays", None) if defense is not None else None

            off_ppd = _safe_div(off_total_ppa, off_drives)
            def_ppd = _safe_div(def_total_ppa, def_drives)
            off_ypp_adj = None
            def_ypp_adj = None
            if off_ppa is not None:
                off_ypp_adj = float(off_ppa) / _EXPECTED_POINTS_PER_YARD
            if def_ppa is not None:
                def_ypp_adj = float(def_ppa) / _EXPECTED_POINTS_PER_YARD

            rows.append(
                {
                    "team": team,
                    "advanced_off_success_rate": off_success,
                    "advanced_off_ppd": off_ppd,
                    "advanced_off_ppa": off_ppa,
                    "advanced_off_ypp": off_ypp_adj,
                    "advanced_off_plays": off_plays,
                    "advanced_off_drives": off_drives,
                    "advanced_def_success_rate": def_success,
                    "advanced_def_ppd": def_ppd,
                    "advanced_def_ppa": def_ppa,
                    "advanced_def_ypp": def_ypp_adj,
                    "advanced_def_plays": def_plays,
                    "advanced_def_drives": def_drives,
                }
            )
        advanced_df = pd.DataFrame(rows)
        if not advanced_df.empty:
            store = advanced_df.copy()
            store["season"] = year
            store["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
            delete_rows(_ADVANCED_DATASET, "season", year)
            storage_write_dataset(store, _ADVANCED_DATASET, if_exists="append")
    if advanced_df.empty:
        return pd.DataFrame(columns=["team"])

    metrics = {
        "advanced_off_success_rate": False,
        "advanced_off_ppd": False,
        "advanced_off_ppa": False,
        "advanced_off_ypp": False,
        "advanced_def_success_rate": True,
        "advanced_def_ppd": True,
        "advanced_def_ppa": True,
        "advanced_def_ypp": True,
    }
    for col, invert in metrics.items():
        scaled_col = col.replace("advanced_", "stat_")
        advanced_df[scaled_col] = _scale_series(advanced_df[col], invert=invert)

    # For downstream merges ensure every stat_* column exists even if NaN
    for col in list(metrics.keys()):
        stat_col = col.replace("advanced_", "stat_")
        if stat_col not in advanced_df.columns:
            advanced_df[stat_col] = None

    return features


__all__ = ["build_team_stat_features"]
