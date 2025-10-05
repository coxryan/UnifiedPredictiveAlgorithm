import pandas as pd
import pytest

from types import SimpleNamespace

import agents.collect.markets as markets
from agents.collect.markets import (
    _combine_market_sources,
    _cfbd_line_to_home_spread,
    _normalize_market_df,
)
import agents.storage as storage
import agents.storage.sqlite_store as sqlite_store


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


def _patch_data_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_upa.sqlite"
    monkeypatch.setattr(sqlite_store, "DATA_DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(storage, "DATA_DB_PATH", str(db_path), raising=False)
    return db_path


def test_get_market_lines_prefers_cached_fanduel(monkeypatch, tmp_path):
    _patch_data_db(monkeypatch, tmp_path)
    monkeypatch.setattr(markets, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(markets, "MARKET_SOURCE", "fanduel", raising=False)
    monkeypatch.setattr(markets, "ODDS_API_KEY", "token", raising=False)

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("should not fetch FanDuel when cache available")

    monkeypatch.setattr(markets, "get_market_lines_fanduel_for_weeks", fail_fetch, raising=False)

    status_updates = []
    monkeypatch.setattr(
        markets,
        "_upsert_status_market_source",
        lambda used, req, reason, *_args, **_kwargs: status_updates.append((used, req, reason)),
        raising=False,
    )

    raw_fanduel = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 5,
                "game_id": 10,
                "home_team": "Home",
                "away_team": "Away",
                "spread": -3.5,
                "retrieved_at": "2025-10-05T00:00:00Z",
            }
        ]
    )
    storage.write_dataset(raw_fanduel, "raw_fanduel_lines")

    schedule = pd.DataFrame(
        [
            {"game_id": 10, "week": 5, "home_team": "Home", "away_team": "Away"}
        ]
    )

    result = markets.get_market_lines_for_current_week(
        2025,
        5,
        schedule,
        apis=SimpleNamespace(lines_api=None),
        cache=SimpleNamespace(),
    )

    assert not result.empty
    assert result.loc[result["game_id"] == 10, "market_spread_fanduel"].iloc[0] == pytest.approx(-3.5)
    assert status_updates[-1][0] == "fanduel"
    assert status_updates[-1][1] == "fanduel"
    assert status_updates[-1][2] in ("", None)


def test_get_market_lines_falls_back_to_cached_cfbd(monkeypatch, tmp_path):
    _patch_data_db(monkeypatch, tmp_path)
    monkeypatch.setattr(markets, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(markets, "MARKET_SOURCE", "fanduel", raising=False)
    monkeypatch.setattr(markets, "ODDS_API_KEY", "token", raising=False)

    def empty_fanduel_fetch(year, weeks, sched_df, cache):
        cols = ["game_id", "week", "home_team", "away_team", "spread"]
        return pd.DataFrame(columns=cols), {"raw": 0, "mapped": 0, "unmatched": 0}

    monkeypatch.setattr(markets, "get_market_lines_fanduel_for_weeks", empty_fanduel_fetch, raising=False)

    status_updates = []
    monkeypatch.setattr(
        markets,
        "_upsert_status_market_source",
        lambda used, req, reason, *_args, **_kwargs: status_updates.append((used, req, reason)),
        raising=False,
    )

    raw_cfbd = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 5,
                "game_id": 20,
                "home_team": "Home",
                "away_team": "Away",
                "spread": -7.0,
                "retrieved_at": "2025-10-05T00:00:00Z",
            }
        ]
    )
    storage.write_dataset(raw_cfbd, "raw_cfbd_lines")

    schedule = pd.DataFrame(
        [
            {"game_id": 20, "week": 5, "home_team": "Home", "away_team": "Away"}
        ]
    )

    result = markets.get_market_lines_for_current_week(
        2025,
        5,
        schedule,
        apis=SimpleNamespace(lines_api=None),
        cache=SimpleNamespace(),
    )

    assert not result.empty
    assert result.loc[result["game_id"] == 20, "market_spread_cfbd"].iloc[0] == pytest.approx(-7.0)
    assert status_updates[-1][0] == "cfbd"
    assert status_updates[-1][1] == "fanduel"
    assert status_updates[-1][2] == "FanDuel available but returned too few rows (0)"


def test_fanduel_nan_logging_includes_raw(monkeypatch, tmp_path):
    logs: list[str] = []
    monkeypatch.setattr(markets, "_dbg", lambda msg: logs.append(msg), raising=False)
    monkeypatch.setattr(markets, "DEBUG_MARKET", True, raising=False)
    monkeypatch.setattr(markets, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(markets, "MARKET_SOURCE", "fanduel", raising=False)
    monkeypatch.setattr(markets, "ODDS_API_KEY", "token", raising=False)

    monkeypatch.setattr(markets, "read_dataset", lambda *_args, **_kwargs: pd.DataFrame(), raising=False)
    monkeypatch.setattr(markets, "storage_write_dataset", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(markets, "delete_rows", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(markets, "get_odds_cache", lambda: None, raising=False)

    fanduel_raw = pd.DataFrame(
        [
            {
                "game_id": 30,
                "week": 5,
                "home_team": "Home",
                "away_team": "Away",
                "spread": None,
                "provider": "FanDuel",
            }
        ]
    )

    def bad_fanduel_fetch(year, weeks, sched_df, cache):
        return fanduel_raw.copy(), {"raw": 1, "mapped": 0, "unmatched": 1}

    monkeypatch.setattr(markets, "get_market_lines_fanduel_for_weeks", bad_fanduel_fetch, raising=False)

    cfbd_df = pd.DataFrame(
        [
            {"game_id": 30, "week": 5, "home_team": "Home", "away_team": "Away", "spread": -6.5}
        ]
    )

    monkeypatch.setattr(markets, "_fetch_cfbd_lines", lambda *args, **kwargs: cfbd_df.copy(), raising=False)

    status_updates = []
    monkeypatch.setattr(
        markets,
        "_upsert_status_market_source",
        lambda used, req, reason, *_args, **_kwargs: status_updates.append((used, req, reason)),
        raising=False,
    )

    schedule = pd.DataFrame(
        [
            {"game_id": 30, "week": 5, "home_team": "Home", "away_team": "Away"}
        ]
    )

    result = markets.get_market_lines_for_current_week(
        2025,
        5,
        schedule,
        apis=SimpleNamespace(lines_api=None),
        cache=SimpleNamespace(),
    )

    assert not result.empty
    assert result.loc[result["game_id"] == 30, "market_spread_cfbd"].iloc[0] == pytest.approx(-6.5)
    assert any("fanduel_missing_detail" in msg for msg in logs)
    detail_line = next(msg for msg in logs if "fanduel_missing_detail" in msg)
    assert "'provider': 'FanDuel'" in detail_line or '"provider": "FanDuel"' in detail_line
