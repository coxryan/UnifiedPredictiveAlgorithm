import pandas as pd

from agents.collect.markets import _combine_market_sources


def test_combine_market_sources_adds_missing_games():
    fanduel = pd.DataFrame(
        [
            {"game_id": 1, "week": 4, "home_team": "Home", "away_team": "Away", "spread": -3.5}
        ]
    )
    cfbd = pd.DataFrame(
        [
            {"game_id": 2, "week": 4, "home_team": "CFBD", "away_team": "Visitor", "market_spread_book": 7.0}
        ]
    )

    combined = _combine_market_sources(fanduel, cfbd)
    assert len(combined) == 2
    assert combined.loc[combined["game_id"] == 1, "spread"].iloc[0] == -3.5
    assert combined.loc[combined["game_id"] == 2, "spread"].iloc[0] == 7.0


def test_combine_market_sources_prefers_non_null_spread():
    fanduel = pd.DataFrame(
        [
            {"game_id": 3, "week": 4, "home_team": "Team A", "away_team": "Team B", "spread": float("nan")}
        ]
    )
    cfbd = pd.DataFrame(
        [
            {"game_id": 3, "week": 4, "home_team": "Team A", "away_team": "Team B", "market_spread_book": -6.0}
        ]
    )

    combined = _combine_market_sources(fanduel, cfbd)
    assert len(combined) == 1
    assert combined.loc[0, "spread"] == -6.0
