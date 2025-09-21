import pandas as pd
from agents.collect.predictions import build_predictions_for_year
from agents.collect import ApiCache


def _team_inputs():
    return pd.DataFrame(
        {
            "team": ["Home", "Away"],
            "wrps_percent_0_100": [60, 50],
            "talent_score_0_100": [62, 48],
            "srs_score_0_100": [65, 45],
        }
    )


def _schedule():
    return pd.DataFrame(
        [
            {
                "game_id": 1,
                "week": 1,
                "home_team": "Home",
                "away_team": "Away",
                "date": "2025-08-24",
                "neutral_site": 0,
            }
        ]
    )


def test_predictions_use_market_when_available(tmp_path):
    schedule_df = _schedule()
    markets = pd.DataFrame(
        [
            {
                "game_id": 1,
                "week": 1,
                "home_team": "Home",
                "away_team": "Away",
                "spread": -3.5,
            }
        ]
    )

    preds = build_predictions_for_year(
        2025,
        schedule_df,
        apis=None,
        cache=ApiCache(root=str(tmp_path / "cache")),
        markets_df=markets,
        team_inputs_df=_team_inputs(),
    )

    assert float(preds.loc[0, "market_spread_book"]) == -3.5
    assert int(preds.loc[0, "market_is_synthetic"]) == 0
    assert preds.loc[0, "edge_points_book"] != 0


def test_predictions_mark_synthetic_when_market_missing(tmp_path):
    schedule_df = _schedule()

    preds = build_predictions_for_year(
        2025,
        schedule_df,
        apis=None,
        cache=ApiCache(root=str(tmp_path / "cache")),
        markets_df=pd.DataFrame(),
        team_inputs_df=_team_inputs(),
    )

    assert pd.isna(preds.loc[0, "market_spread_book"])
    assert int(preds.loc[0, "market_is_synthetic"]) == 1
