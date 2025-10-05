from __future__ import annotations

import os

import pandas as pd

from agents.collect import (
    CfbdClients,
    ApiCache,
    build_team_stat_features,
    build_team_inputs_datadriven,
)
from agents.storage import write_dataset as storage_write_dataset, delete_rows, read_dataset


def main() -> None:
    year = int(os.environ.get("YEAR", pd.Timestamp.utcnow().year))
    bearer = os.environ.get("CFBD_BEARER_TOKEN", "").strip()

    apis = CfbdClients(bearer_token=bearer)
    cache = ApiCache()

    features = build_team_stat_features(year, apis, cache)
    if not features.empty:
        store_df = features.copy()
        store_df["season"] = year
        store_df["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
        delete_rows("raw_cfbd_team_stats_features", "season", year)
        storage_write_dataset(store_df, "raw_cfbd_team_stats_features", if_exists="append")

    team_inputs = build_team_inputs_datadriven(year, apis, cache)
    team_inputs["season"] = year
    delete_rows("upa_team_inputs_datadriven_v0", "season", year)
    storage_write_dataset(team_inputs, "upa_team_inputs_datadriven_v0", if_exists="append")


if __name__ == "__main__":
    main()
