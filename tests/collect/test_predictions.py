import pandas as pd
import pytest
from agents.collect.predictions import build_predictions_for_year, _rating_from_team_inputs
from agents.collect import ApiCache


def _team_inputs():
    return pd.DataFrame(
        {
            "team": ["Home", "Away"],
            "wrps_percent_0_100": [60, 50],
            "talent_score_0_100": [62, 48],
            "srs_score_0_100": [65, 45],
            "stat_off_index_0_100": [70, 40],
            "stat_def_index_0_100": [68, 42],
            "stat_st_index_0_100": [55, 45],
            "grade_qb_score": [72, 38],
            "grade_qb_letter": ["B-", "D"],
            "grade_wr_score": [75, 35],
            "grade_wr_letter": ["C+", "D-"],
            "grade_rb_score": [70, 40],
            "grade_rb_letter": ["C-", "D+"],
            "grade_ol_score": [68, 42],
            "grade_ol_letter": ["D+", "D"],
            "grade_dl_score": [73, 37],
            "grade_dl_letter": ["C", "D-"],
            "grade_lb_score": [71, 39],
            "grade_lb_letter": ["C-", "D"],
            "grade_db_score": [74, 36],
            "grade_db_letter": ["C", "D-"],
            "grade_st_score": [66, 34],
            "grade_st_letter": ["D+", "F"],
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
    assert "home_grade_qb_letter" in preds.columns
    assert preds.loc[0, "home_grade_qb_letter"] != ""
    assert "market_adjustment" in preds.columns
    assert float(preds.loc[0, "market_adjustment"]) == pytest.approx(0.0)
    assert "model_confidence" in preds.columns


def test_predictions_sets_market_source_cfbd(tmp_path):
    schedule_df = _schedule()
    markets = pd.DataFrame(
        [
            {
                "game_id": 1,
                "week": 1,
                "home_team": "Home",
                "away_team": "Away",
                "spread": -6.5,
                "market_spread_cfbd": -6.5,
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

    assert preds.loc[0, "market_spread_source"] == "cfbd"
    assert int(preds.loc[0, "market_is_synthetic"]) == 0
    assert float(preds.loc[0, "market_adjustment"]) == pytest.approx(0.0)


def test_rating_from_team_inputs_uses_stat_features():
    df = pd.DataFrame(
        {
            "team": ["A", "B"],
            "wrps_percent_0_100": [50, 50],
            "talent_score_0_100": [50, 50],
            "srs_score_0_100": [50, 50],
            "stat_off_index_0_100": [80, 20],
            "stat_def_index_0_100": [75, 25],
            "stat_st_index_0_100": [60, 40],
        }
    )
    ratings = _rating_from_team_inputs(df)
    assert ratings.loc["A"] > ratings.loc["B"]


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
    assert float(preds.loc[0, "market_adjustment"]) == pytest.approx(0.0)


def test_market_selection_logs_raw_when_fanduel_nan(monkeypatch, caplog, tmp_path):
    schedule_df = _schedule()
    markets = pd.DataFrame(
        [
            {
                "game_id": 1,
                "week": 1,
                "home_team": "Home",
                "away_team": "Away",
                "market_spread_fanduel": "PK",
                "spread": -3.5,
            }
        ]
    )

    monkeypatch.setenv("DEBUG_MARKET", "1")

    with caplog.at_level("DEBUG"):
        build_predictions_for_year(
            2025,
            schedule_df,
            apis=None,
            cache=ApiCache(root=str(tmp_path / "cache")),
            markets_df=markets,
            team_inputs_df=_team_inputs(),
        )

    messages = [msg for msg in caplog.messages if msg.startswith("market selection:")]
    assert any("fan_duel_raw='PK'" in msg for msg in messages)
