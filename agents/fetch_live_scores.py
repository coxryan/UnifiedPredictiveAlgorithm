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
            t = side.get("team") or {}
            display = t.get("displayName") or t.get("shortDisplayName") or t.get("name")
            nickname = t.get("name") or ""
            location = t.get("location") or ""
            # Derive a CFBD-style school name: prefer 'location'; else strip trailing nickname
            school = location or display
            if display and nickname and display.endswith(" " + nickname):
                school = display[: -(len(nickname) + 1)].strip()
            # Score
            score = side.get("score")
            try:
                score = int(score) if score is not None else None
            except Exception:
                score = None
            return {
                "display": display,
                "school": school,
                "nickname": nickname,
                "score": score,
            }

        h = team(home)
        a = team(away)

        rows.append({
            "event_id": ev.get("id"),
            "date": ev.get("date"),
            "state": state,
            "detail": detail,
            "clock": clock,
            "period": period,
            "venue": venue,
            # ESPN raw display
            "home_team": h["display"],
            "away_team": a["display"],
            # Normalized school names to match CFBD/our CSVs
            "home_school": h["school"],
            "away_school": a["school"],
            # Scores
            "home_points": h["score"],
            "away_points": a["score"],
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
    # Ensure output directory exists when a folder is provided.
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} rows")

if __name__ == "__main__":
    main()