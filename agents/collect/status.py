from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from .config import DATA_DIR


def _upsert_status_market_source(
    market_used: str,
    market_requested: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    data_dir: str = DATA_DIR,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Merge-update data/status.json with the selected market source and status fields.
    Always writes:
      - market_source_used: the used source (lowercased)
      - market_source: (back-compat) the used source (lowercased)
      - market_source_config/requested_market: requested source (lowercased, if provided)
      - fallback_reason/market_fallback_reason: string if provided
      - generated_at_utc: current UTC ISO timestamp
    Keeps other fields intact if present; creates the file if missing.
    """
    try:
        os.makedirs(data_dir, exist_ok=True)
        p = os.path.join(data_dir, "status.json")
        payload: Dict[str, Any]
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    payload = json.load(f) or {}
            except Exception:
                payload = {}
        else:
            payload = {}

        used_lc = (market_used or "cfbd").strip().lower()
        payload["market_source_used"] = used_lc
        payload["market_source"] = used_lc  # back-compat

        if market_requested is not None:
            req_lc = (market_requested or "").strip().lower()
            payload["market_source_config"] = req_lc
            payload["requested_market"] = req_lc
            payload["market_requested"] = req_lc
            payload["market_source_requested"] = req_lc
        else:
            for k in ("market_source_config", "requested_market", "market_requested", "market_source_requested"):
                payload.pop(k, None)

        if fallback_reason:
            payload["market_fallback_reason"] = str(fallback_reason)
            payload["fallback_reason"] = str(fallback_reason)
        else:
            payload.pop("market_fallback_reason", None)
            payload.pop("fallback_reason", None)

        payload["generated_at_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        if extra:
            payload.update(extra)

        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        # never crash on status writing
        pass


__all__ = ["_upsert_status_market_source"]

