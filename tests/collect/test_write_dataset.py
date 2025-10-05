import pandas as pd

from agents.collect.helpers import write_dataset
from agents.storage import sqlite_store
from agents.storage import write_dataset as storage_write_dataset, read_dataset


def test_write_dataset_backfills_from_market_debug(tmp_path):
    sqlite_store.DATA_DB_PATH = str(tmp_path / "data.sqlite")
    market_debug = pd.DataFrame(
        [
            {
                "game_id": 10,
                "week": 5,
                "home_team": "Home",
                "away_team": "Away",
                "market_spread_book": -7.0,
            }
        ]
    )
    storage_write_dataset(market_debug, "market_debug")

    df = pd.DataFrame(
        [
            {
                "game_id": 10,
                "week": 5,
                "home_team": "Home",
                "away_team": "Away",
                "model_spread_book": -6.5,
                "market_spread_book": float("nan"),
                "market_is_synthetic": 0,
            }
        ]
    )

    write_dataset(df.copy(), "upa_predictions")

    result = read_dataset("upa_predictions")
    assert float(result.loc[0, "market_spread_book"]) == -7.0
    assert int(result.loc[0, "market_is_synthetic"]) == 0


def test_write_dataset_marks_synthetic_when_still_missing(tmp_path):
    sqlite_store.DATA_DB_PATH = str(tmp_path / "data.sqlite")
    df = pd.DataFrame(
        [
            {
                "game_id": 11,
                "week": 5,
                "home_team": "Home",
                "away_team": "Away",
                "model_spread_book": -4.0,
                "market_spread_book": float("nan"),
                "market_is_synthetic": 0,
            }
        ]
    )

    write_dataset(df.copy(), "upa_predictions")

    result = read_dataset("upa_predictions")
    assert int(result.loc[0, "market_is_synthetic"]) == 1
