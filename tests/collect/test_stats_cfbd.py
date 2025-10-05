import pandas as pd

from agents.collect.stats_cfbd import _prepare_feature_frame


def test_prepare_feature_frame_scales_and_inverts():
    stats_rows = pd.DataFrame(
        [
            {"team": "Team A", "stat_name": "offense.points_per_game", "stat_value": 40.0},
            {"team": "Team B", "stat_name": "offense.points_per_game", "stat_value": 20.0},
            {"team": "Team A", "stat_name": "defense.points_per_game", "stat_value": 18.0},
            {"team": "Team B", "stat_name": "defense.points_per_game", "stat_value": 35.0},
            {"team": "Team A", "stat_name": "offense.yards_per_play", "stat_value": 7.5},
            {"team": "Team B", "stat_name": "offense.yards_per_play", "stat_value": 5.0},
            {"team": "Team A", "stat_name": "defense.yards_per_play", "stat_value": 4.8},
            {"team": "Team B", "stat_name": "defense.yards_per_play", "stat_value": 6.2},
        ]
    )

    frame = _prepare_feature_frame(stats_rows)
    assert set(frame.columns) >= {"team", "stat_off_ppg", "stat_def_ppg", "stat_off_ypp", "stat_def_ypp"}
    row_a = frame.loc[frame["team"] == "Team A"].iloc[0]
    row_b = frame.loc[frame["team"] == "Team B"].iloc[0]
    assert row_a["stat_off_ppg"] > row_b["stat_off_ppg"]
    assert row_a["stat_def_ppg"] > row_b["stat_def_ppg"]  # inverted: lower ppg better
    assert 0.0 <= row_a["stat_off_ppg"] <= 100.0
    assert 0.0 <= row_b["stat_def_ypp"] <= 100.0
