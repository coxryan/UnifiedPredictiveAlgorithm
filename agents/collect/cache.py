from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from .config import ODDS_CACHE_DIR, ODDS_CACHE_TTL_DAYS, CACHE_DIR, CACHE_TTL_DAYS
from agents.storage.sqlite_store import (
    write_cache_entry,
    read_cache_entry,
    purge_cache_entry,
)


class ApiCache:
    """SQLite-backed cache with TTL semantics."""

    def __init__(self, root: str = CACHE_DIR, days_to_live: int = CACHE_TTL_DAYS):
        self.ttl = max(1, int(days_to_live)) * 86400
        root = root or CACHE_DIR
        os.makedirs(root, exist_ok=True)
        self.db_path = os.path.join(root, "cache.sqlite")

    def get(self, key: str) -> Tuple[bool, Any]:
        payload = read_cache_entry(key, db_path=self.db_path)
        return (True, payload) if payload is not None else (False, None)

    def set(self, key: str, value: Any) -> None:
        write_cache_entry(key, value, self.ttl, db_path=self.db_path)

    def purge(self, key: str) -> None:
        purge_cache_entry(key, db_path=self.db_path)


_odds_cache_singleton: Optional[ApiCache] = None


def get_odds_cache() -> ApiCache:
    global _odds_cache_singleton
    if _odds_cache_singleton is None:
        _odds_cache_singleton = ApiCache(root=ODDS_CACHE_DIR, days_to_live=ODDS_CACHE_TTL_DAYS)
    return _odds_cache_singleton


__all__ = ["ApiCache", "get_odds_cache"]
