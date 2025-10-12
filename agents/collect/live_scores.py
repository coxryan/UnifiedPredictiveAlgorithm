from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from agents.fetch_live_scores import fetch_scoreboard
from agents.storage import read_dataset, write_dataset
from agents.storage import write_dataset as storage_write_dataset


LIVE_SCORES_COLUMNS = [
    "event_id",
    "date",
    "state",
    "detail",
    "clock",
    "period",
    "venue",
    "home_team",
    "away_team",
    "home_school",
    "away_school",
    "home_points",
    "away_points",
]


logger = logging.getLogger(__name__)


def _now_pacific() -> pd.Timestamp:
    try:
        import pytz  # type: ignore

        return pd.Timestamp.now(tz=pytz.timezone("America/Los_Angeles"))
    except Exception:
        return pd.Timestamp.utcnow()


def _build_date_strings(days: int) -> Iterable[str]:
    base = _now_pacific()
    span = max(1, int(days))
    return sorted({(base - pd.Timedelta(days=offset)).strftime("%Y%m%d") for offset in range(span)})


def update_live_scores(year: int, *, days: int = 3) -> pd.DataFrame:
    """Fetch ESPN scoreboard snapshots for the recent window and persist them.

    Returns the combined `live_scores` dataframe (existing rows preserved, new ones merged).
    """

    try:
        existing = read_dataset("live_scores")
    except Exception:
        existing = pd.DataFrame(columns=LIVE_SCORES_COLUMNS)

    fetched_frames: list[pd.DataFrame] = []
    for day in _build_date_strings(days):
        try:
            rows = fetch_scoreboard(day)
        except Exception as exc:
            logger.warning("update_live_scores: scoreboard fetch failed for %s: %s", day, exc)
            continue
        if not rows:
            continue
        df_day = pd.DataFrame(rows)
        for col in LIVE_SCORES_COLUMNS:
            if col not in df_day.columns:
                df_day[col] = None
        df_day = df_day[LIVE_SCORES_COLUMNS]
        fetched_frames.append(df_day)

    if fetched_frames:
        new_scores = pd.concat(fetched_frames, ignore_index=True)
        combined = pd.concat([existing, new_scores], ignore_index=True)
        if "event_id" in combined.columns:
            combined = combined.drop_duplicates(subset=["event_id"], keep="last")
        write_dataset(combined, "live_scores")

        archived = new_scores.copy()
        archived["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
        archived["season"] = year
        storage_write_dataset(archived, "raw_espn_scoreboard", if_exists="append")

        logger.debug(
            "update_live_scores: appended %s rows (combined=%s)", len(new_scores), len(combined)
        )

        return combined

    # No new data fetched; retain existing snapshot
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        write_dataset(existing, "live_scores")
        logger.debug(
            "update_live_scores: no new rows fetched; retaining %s existing rows",
            len(existing),
        )
        return existing

    empty = pd.DataFrame(columns=LIVE_SCORES_COLUMNS)
    write_dataset(empty, "live_scores")
    logger.debug("update_live_scores: no data available; wrote empty snapshot")
    return empty


__all__ = ["update_live_scores", "LIVE_SCORES_COLUMNS"]
