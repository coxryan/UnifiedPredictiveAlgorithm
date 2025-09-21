from __future__ import annotations

import os
import sys
from typing import Any

# ======================================================
# Config / Env
# ======================================================
DATA_DIR = os.environ.get("DATA_DIR", "data").strip() or "data"
DATA_DB_PATH = os.environ.get("DATA_DB_PATH", os.path.join(DATA_DIR, "upa_data.sqlite"))

# Generic API cache (CFBD + other)
CACHE_DIR = os.environ.get("CFBD_CACHE_DIR", os.environ.get("CACHE_DIR", ".cache_cfbd"))
CACHE_TTL_DAYS = int(os.environ.get("CACHE_TTL_DAYS", "90"))
CACHE_DB_PATH = os.environ.get("CACHE_DB_PATH", os.path.join(DATA_DIR, "upa_cache.sqlite"))

# If set, do not hit any external APIs; use cache only (write empty on miss)
CACHE_ONLY = os.environ.get("CACHE_ONLY", "0").strip().lower() in ("1", "true", "yes")

# Optional Google Sheets push (disabled by default)
ENABLE_SHEETS = False
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
MARKET_SOURCE = os.environ.get("MARKET_SOURCE", "fanduel").strip().lower()

# Separate cache for FanDuel (Odds API) so we can flip sources freely
ODDS_CACHE_DIR = os.environ.get("ODDS_CACHE_DIR", ".cache_odds")
ODDS_CACHE_TTL_DAYS = int(os.environ.get("ODDS_CACHE_TTL_DAYS", "2"))

# Require schedule to have at least N rows (guards against bad/missing reads)
REQUIRE_SCHED_MIN_ROWS = int(os.environ.get("REQUIRE_SCHED_MIN_ROWS", "0") or 0)

# Debug logging for market selection & matching
DEBUG_MARKET = os.environ.get("DEBUG_MARKET", "0").strip().lower() in ("1", "true", "yes", "y")
MARKET_MIN_ROWS = int(os.environ.get("MARKET_MIN_ROWS", "1"))  # minimum rows to consider a market usable


def _dbg(msg: str) -> None:
    if DEBUG_MARKET:
        try:
            print(f"[debug-market] {msg}", file=sys.stderr)
        except Exception:
            pass


__all__ = [
    "DATA_DIR","DATA_DB_PATH","CACHE_DIR","CACHE_TTL_DAYS","CACHE_DB_PATH","CACHE_ONLY","ENABLE_SHEETS","ODDS_API_KEY","MARKET_SOURCE",
    "ODDS_CACHE_DIR","ODDS_CACHE_TTL_DAYS","REQUIRE_SCHED_MIN_ROWS","DEBUG_MARKET","MARKET_MIN_ROWS","_dbg",
]
