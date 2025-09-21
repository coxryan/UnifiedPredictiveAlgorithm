from __future__ import annotations

import os
from datetime import datetime

from agents.collect import (
    CfbdClients,
    ApiCache,
    load_schedule_for_year,
    build_team_inputs_datadriven,
)


def main() -> None:
    year = int(os.environ.get("YEAR", datetime.utcnow().year))
    bearer = os.environ.get("CFBD_BEARER_TOKEN", "").strip()

    apis = CfbdClients(bearer_token=bearer)
    cache = ApiCache()

    # Refresh schedule snapshot so weekly metadata has the latest mappings.
    load_schedule_for_year(year, apis, cache)

    # Build team inputs to refresh returning production, talent, SRS, SOS raw tables.
    build_team_inputs_datadriven(year, apis, cache)


if __name__ == "__main__":
    main()
