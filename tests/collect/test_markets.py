import pandas as pd

from types import SimpleNamespace

from agents.collect.markets import _combine_market_sources, _cfbd_line_to_home_spread, _normalize_market_df


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


def test_cfbd_line_to_home_spread_home_favorite():
    line = SimpleNamespace(
        provider="consensus",
        spread=-7.5,
        formatted_spread="Texas -7.5",
        home_moneyline=-210,
        away_moneyline=175,
    )
    assert _cfbd_line_to_home_spread("Texas", "Oklahoma", line) == -7.5


def test_cfbd_line_to_home_spread_away_favorite():
    line = SimpleNamespace(
        provider="consensus",
        spread=-3.0,
        formatted_spread="Oklahoma -3",
        home_moneyline=155,
        away_moneyline=-180,
    )
    assert _cfbd_line_to_home_spread("Texas", "Oklahoma", line) == 3.0


def test_cfbd_line_to_home_spread_fallback_moneyline():
    line = SimpleNamespace(
        provider="consensus",
        spread=-4.5,
        formatted_spread=None,
        home_moneyline=150,
        away_moneyline=-170,
    )
    assert _cfbd_line_to_home_spread("Texas", "Oklahoma", line) == 4.5


def test_normalize_market_df_drops_nan_spread():
    df = pd.DataFrame(
        [
            {"game_id": 1, "week": 1, "home_team": "A", "away_team": "B", "spread": float("nan")},
            {"game_id": 2, "week": 1, "home_team": "C", "away_team": "D", "spread": -3.5},
        ]
    )
    normalized = _normalize_market_df(df)
    assert len(normalized) == 1
    assert normalized.iloc[0]["game_id"] == 2
