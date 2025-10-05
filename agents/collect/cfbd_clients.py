from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cfbd  # type: ignore
except Exception:  # pragma: no cover - optional import
    cfbd = None  # type: ignore


@dataclass
class CfbdClients:
    bearer_token: str
    teams_api: Any = None
    players_api: Any = None
    ratings_api: Any = None
    games_api: Any = None
    lines_api: Any = None
    rankings_api: Any = None
    stats_api: Any = None

    def __post_init__(self):
        if cfbd and self.bearer_token:
            cfg = cfbd.Configuration(access_token=self.bearer_token)
            cli = cfbd.ApiClient(cfg)
            self.teams_api = cfbd.TeamsApi(cli)
            self.players_api = cfbd.PlayersApi(cli)
            self.ratings_api = cfbd.RatingsApi(cli)
            self.games_api = cfbd.GamesApi(cli)
            self.lines_api = cfbd.BettingApi(cli) if hasattr(cfbd, 'BettingApi') else cfbd.LinesApi(cli)
            self.rankings_api = cfbd.RankingsApi(cli)
            if hasattr(cfbd, 'StatsApi'):
                self.stats_api = cfbd.StatsApi(cli)


__all__ = ["CfbdClients"]
