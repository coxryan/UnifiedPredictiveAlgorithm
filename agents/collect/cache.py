from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional, Tuple

from .config import ODDS_CACHE_DIR, ODDS_CACHE_TTL_DAYS, CACHE_DIR, CACHE_TTL_DAYS


class ApiCache:
    """Simple JSON file-backed cache with time-based expiry."""

    def __init__(self, root: str = CACHE_DIR, days_to_live: int = CACHE_TTL_DAYS):
        self.root = root
        self.ttl = max(1, int(days_to_live)) * 86400
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        sub = os.path.join(self.root, h[:2], h[2:4])
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{h}.json")

    def get(self, key: str) -> Tuple[bool, Any]:
        p = self._path(key)
        if not os.path.exists(p):
            return False, None
        try:
            if (time.time() - os.path.getmtime(p)) > self.ttl:
                return False, None
            with open(p, "r") as f:
                return True, json.load(f)
        except Exception:
            return False, None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        with open(p, "w") as f:
            json.dump(value, f)


_odds_cache_singleton: Optional[ApiCache] = None


def get_odds_cache() -> ApiCache:
    global _odds_cache_singleton
    if _odds_cache_singleton is None:
        _odds_cache_singleton = ApiCache(root=ODDS_CACHE_DIR, days_to_live=ODDS_CACHE_TTL_DAYS)
    return _odds_cache_singleton


__all__ = ["ApiCache", "get_odds_cache"]

