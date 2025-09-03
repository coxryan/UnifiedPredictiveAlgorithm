# agents/lib/cfbd_clients.py
import os, sys
try:
    import cfbd
except Exception:
    print("ERROR: cfbd not installed. pip install cfbd", file=sys.stderr)
    raise

def build_clients(bearer_token: str):
    cfg = cfbd.Configuration(access_token=bearer_token)
    api_client = cfbd.ApiClient(cfg)
    return dict(
        teams=cfbd.TeamsApi(api_client),
        players=cfbd.PlayersApi(api_client),
        ratings=cfbd.RatingsApi(api_client),
        games=cfbd.GamesApi(api_client),
        betting=cfbd.BettingApi(api_client),
        raw_client=api_client,
    )