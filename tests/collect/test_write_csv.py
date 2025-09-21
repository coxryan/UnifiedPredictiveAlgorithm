import pandas as pd
from agents.collect.helpers import write_csv


def test_write_csv_backfills_from_market_debug(tmp_path):
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
    market_debug.to_csv(data_dir / "market_debug.csv", index=False)

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

    result = pd.read_csv(out_path)
    assert float(result.loc[0, "market_spread_book"]) == -7.0
    assert int(result.loc[0, "market_is_synthetic"]) == 0


def test_write_csv_marks_synthetic_when_still_missing(tmp_path):
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

    result = pd.read_csv(out_path)
    assert int(result.loc[0, "market_is_synthetic"]) == 1
