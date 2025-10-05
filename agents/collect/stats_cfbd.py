from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from .cache import ApiCache
from .cfbd_clients import CfbdClients
from .config import _dbg
from agents.storage import read_dataset, write_dataset as storage_write_dataset, delete_rows

_STAT_FEATURE_MAP: Dict[Tuple[str, str], str] = {
    ("offense", "pointsPerGame"): "stat_off_ppg",
    ("offense", "yardsPerPlay"): "stat_off_ypp",
    ("offense", "successRate"): "stat_off_success",
    ("offense", "explosiveness"): "stat_off_explosiveness",
    ("defense", "pointsPerGame"): "stat_def_ppg",
    ("defense", "yardsPerPlay"): "stat_def_ypp",
    ("defense", "successRate"): "stat_def_success",
    ("defense", "explosiveness"): "stat_def_explosiveness",
    ("specialTeams", "pointsPerPlay"): "stat_st_points_per_play",
}

_INVERT_METRICS: frozenset[str] = frozenset(
    {
        "stat_def_ppg",
        "stat_def_ypp",
        "stat_def_success",
        "stat_def_explosiveness",
    }
)


def _serialize_stats_payload(payload: Iterable[Any], category: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in payload or []:
        team_name = getattr(item, "team", None) or getattr(item, "school", None)
        conference = getattr(item, "conference", None)
        stats = getattr(item, "stats", [])
        for stat in stats or []:
            stat_name = getattr(stat, "stat_name", None) or getattr(stat, "stat", None)
            stat_value = getattr(stat, "stat_value", None) or getattr(stat, "value", None)
            if team_name and stat_name is not None:
                rows.append(
                    {
                        "team": str(team_name),
                        "conference": str(conference) if conference is not None else None,
                        "category": category,
                        "stat_name": str(stat_name),
                        "stat_value": float(stat_value) if stat_value not in (None, "") else None,
                    }
                )
    return rows


def _fetch_team_stats(year: int, category: str, apis: CfbdClients, cache: ApiCache) -> List[Dict[str, Any]]:
    cache_key = f"team-stats:{year}:{category}"
    ok, cached = cache.get(cache_key)
    if ok and isinstance(cached, list):
        return cached
    if not getattr(apis, "stats_api", None):
        return []
    try:
        payload = apis.stats_api.get_team_season_stats(year=year, season_type="regular", category=category)
    except Exception as exc:  # pragma: no cover - remote failure path
        _dbg(f"team stats fetch failed category={category} err={exc}")
        return []
    rows = _serialize_stats_payload(payload, category)
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
    pivot_sources = []
    for (category, stat_name), feature in _STAT_FEATURE_MAP.items():
        mask = (stats_df["category"] == category) & (stats_df["stat_name"].str.lower() == stat_name.lower())
        if not mask.any():
            continue
        subset = stats_df.loc[mask, ["team", "stat_value"]].copy()
        subset = subset.rename(columns={"stat_value": feature})
        pivot_sources.append(subset)
    if not pivot_sources:
        return pd.DataFrame(columns=["team"] + list(_STAT_FEATURE_MAP.values()))
    merged = pivot_sources[0]
    for df_piece in pivot_sources[1:]:
        merged = merged.merge(df_piece, on="team", how="outer")
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
        categories = sorted({key[0] for key in _STAT_FEATURE_MAP.keys()})
        rows: List[Dict[str, Any]] = []
        for cat in categories:
            rows.extend(_fetch_team_stats(year, cat, apis, cache))
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
    return features


__all__ = ["build_team_stat_features"]
