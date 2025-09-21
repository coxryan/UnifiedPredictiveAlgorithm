import pandas as pd

from agents.collect.helpers import write_csv
from agents.storage import sqlite_store
from agents.storage.sqlite_store import write_table_from_path, read_table_from_path


def test_write_csv_backfills_from_market_debug(tmp_path):
    sqlite_store.DATA_DB_PATH = str(tmp_path / "data.sqlite")
    data_dir = tmp_path
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
    write_table_from_path(market_debug, str(data_dir / "market_debug.csv"))

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

    out_path = data_dir / "upa_predictions.csv"
    write_csv(df.copy(), str(out_path))

    result = read_table_from_path(str(out_path))
    assert float(result.loc[0, "market_spread_book"]) == -7.0
    assert int(result.loc[0, "market_is_synthetic"]) == 0


def test_write_csv_marks_synthetic_when_still_missing(tmp_path):
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

    out_path = tmp_path / "upa_predictions.csv"
    write_csv(df.copy(), str(out_path))

    result = read_table_from_path(str(out_path))
    assert int(result.loc[0, "market_is_synthetic"]) == 1
