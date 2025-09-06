from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Dict, Any
import os

import pandas as pd
import requests


def _get_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_scoreboard(day: str | None = None) -> List[Dict[str, Any]]:
    """
    Pull ESPN’s public scoreboard JSON for a given UTC day (YYYYMMDD).
    If None, uses today (UTC).
    """
    if not day:
        day = dt.datetime.utcnow().strftime("%Y%m%d")

    url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates={day}"
    data = _get_json(url)
    events = data.get("events") or []

    rows: List[Dict[str, Any]] = []
    for ev in events:
        comps = (ev.get("competitions") or [])
        if not comps:
            continue
        c = comps[0]

        st = (c.get("status") or {}).get("type") or {}
        state = st.get("state")          # "pre" | "in" | "post"
        detail = st.get("detail")        # "10:44 - 4th", "Final", etc.
        clock = (c.get("status") or {}).get("displayClock")
        period = (c.get("status") or {}).get("period")
        venue = (c.get("venue") or {}).get("fullName")

        competitors = c.get("competitors") or []
        home = next((t for t in competitors if t.get("homeAway") == "home"), None)
        away = next((t for t in competitors if t.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        def team(side):
            team = side.get("team") or {}
            name = team.get("displayName") or team.get("name") or team.get("shortDisplayName")
            score = side.get("score")
            try:
                score = int(score) if score is not None else None
            except Exception:
                score = None
            return name, score

        home_team, home_points = team(home)
        away_team, away_points = team(away)

        rows.append({
            "event_id": ev.get("id"),
            "date": ev.get("date"),
            "state": state,
            "detail": detail,
            "clock": clock,
            "period": period,
            "venue": venue,
            "home_team": home_team,
            "home_points": home_points,
            "away_team": away_team,
            "away_points": away_points,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=dt.datetime.utcnow().year)
    p.add_argument("--out", type=str, default="data/live_scores.csv")
    p.add_argument("--day", type=str, help="YYYYMMDD (UTC)")
    args = p.parse_args()

    rows = fetch_scoreboard(args.day)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} rows")

if __name__ == "__main__":
    main()