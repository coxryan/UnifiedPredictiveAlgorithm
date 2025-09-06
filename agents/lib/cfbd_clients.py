from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import cfbd

from agents.lib.cache import ApiCache


class CfbdClients:
    """
    Wrapper around cfbd SDK clients, with lightweight DataFrame helpers.
    Provides caching for expensive calls (schedule, lines, etc.).
    """

    def __init__(self, bearer: str, cache: ApiCache):
        configuration = cfbd.Configuration(access_token=bearer) if bearer else cfbd.Configuration()
        self.api_client = cfbd.ApiClient(configuration)
        self.teams_api = cfbd.TeamsApi(self.api_client)
        self.players_api = cfbd.PlayersApi(self.api_client)
        self.ratings_api = cfbd.RatingsApi(self.api_client)
        self.games_api = cfbd.GamesApi(self.api_client)
        self.lines_api = cfbd.LinesApi(self.api_client)

        self.cache = cache

    # ------------------------------
    # Cached helpers
    # ------------------------------
    def get_schedule_df(self, season: int) -> pd.DataFrame:
        key = ("schedule", season)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        try:
            games = self.games_api.get_games(year=season, season_type="both")
        except Exception as e:
            print(f"[warn] schedule fetch failed for {season}: {e}")
            return pd.DataFrame()

        rows = []
        for g in games or []:
            rows.append({
                "game_id": getattr(g, "id", None),
                "season": season,
                "week": getattr(g, "week", None),
                "date": getattr(g, "start_date", None),
                "home_team": getattr(g, "home_team", None),
                "away_team": getattr(g, "away_team", None),
                "neutral_site": int(bool(getattr(g, "neutral_site", False))),
            })

        df = pd.DataFrame(rows)
        self.cache.set(key, df, ttl=86400)  # 1 day
        return df

    def get_market_lines_df(self, season: int, week: int) -> pd.DataFrame:
        """
        Fetch market lines for given season+week from cfbd.LinesApi.
        Shape: game_id, week, home_team, away_team, spread
        """
        key = ("market_lines", season, week)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        try:
            lines = self.lines_api.get_lines(year=season, week=week)
        except Exception as e:
            print(f"[warn] market lines fetch failed for {season} w{week}: {e}")
            return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"])

        rows = []
        for ln in lines or []:
            game_id = getattr(ln, "id", None)
            week = getattr(ln, "week", None)
            ht = getattr(ln, "home_team", None)
            at = getattr(ln, "away_team", None)

            # line spreads: cfbd schema has lines[] per provider; take consensus/first
            spread = None
            try:
                if hasattr(ln, "lines") and ln.lines:
                    # lines is list of cfbd.models.Line
                    for l in ln.lines:
                        if hasattr(l, "spread") and l.spread is not None:
                            spread = l.spread
                            break
            except Exception:
                pass

            if spread is not None:
                rows.append({
                    "game_id": game_id,
                    "week": week,
                    "home_team": ht,
                    "away_team": at,
                    "spread": spread,
                })

        df = pd.DataFrame(rows)
        self.cache.set(key, df, ttl=3600)  # cache 1h
        return df