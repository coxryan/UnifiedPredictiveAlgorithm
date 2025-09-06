from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .cfbd_clients import CfbdClients
from .cache import ApiCache

__all__ = ["build_predictions_for_year"]


# ----------------------------
# Helpers
# ----------------------------
def _hf_adjust(neutral: pd.Series) -> pd.Series:
    """Home-field advantage (points) if not neutral."""
    # Simple constant HFA; feel free to calibrate later
    return np.where(neutral.astype(int) == 1, 0.0, 2.0)


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _team_rating(df_teams: pd.DataFrame) -> pd.Series:
    """
    Build a composite 0..100 rating for each team using available inputs.
    You can tune weights as you calibrate.
    """
    # Pull components; coerce to numeric
    wrps = pd.to_numeric(df_teams.get("wrps_percent_0_100"), errors="coerce")
    talent = pd.to_numeric(df_teams.get("talent_score_0_100"), errors="coerce")
    portal = pd.to_numeric(df_teams.get("portal_net_0_100"), errors="coerce")

    # Fill missing with medians to keep the model defined
    wrps = wrps.fillna(wrps.median() if not wrps.dropna().empty else 50.0)
    talent = talent.fillna(talent.median() if not talent.dropna().empty else 50.0)
    portal = portal.fillna(portal.median() if not portal.dropna().empty else 50.0)

    # Weights (tune later)
    w_wrps = 0.5
    w_talent = 0.3
    w_portal = 0.2

    rating = w_wrps * wrps + w_talent * talent + w_portal * portal
    # Keep in a reasonable 0..100 band
    return rating.clip(lower=0.0, upper=100.0)


def _home_spread_from_ratings(home_r: float, away_r: float, neutral_flag: int) -> float:
    """
    Convert rating delta to points; add HFA when not neutral.
    Scaling factor is a quick heuristic; you can re-fit from historical data.
    """
    # 12 rating points ~ 1 TD (tunable)
    scale = 12.0
    hfa = 0.0 if neutral_flag == 1 else 2.0
    return (home_r - away_r) / (scale / 7.0) + hfa


def _book_style(spread_home: pd.Series) -> pd.Series:
    """
    Make spread "book-style": negative favors home, positive favors away.
    (If your internal is already home-positive, multiply by -1.)
    """
    return -1.0 * spread_home


# ----------------------------
# Main builder
# ----------------------------
def build_predictions_for_year(
    season: int,
    apis: CfbdClients,
    cache: ApiCache,
    schedule_df: pd.DataFrame,
    team_inputs_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame],
    current_week: Optional[int],
    force_book_style: bool = True,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns used by the UI:
      week, date, away, home, neutral, model_h, market_h, expected_h, edge, value, qualified, game_id
    Notes:
      - model_h, market_h, expected_h are *home* spreads (home-positive internally).
      - If `force_book_style` True, the returned columns are converted to book-style (home favorites negative).
      - Market is only used for current_week; other weeks use 0 as requested.
    """

    # 1) Build per-team ratings
    teams = team_inputs_df.copy()
    teams["rating_0_100"] = _team_rating(teams)

    # 2) Prepare schedule (ensure expected columns)
    sched = schedule_df.copy()
    for col in ["game_id", "week", "date", "home_team", "away_team", "neutral_site"]:
        if col not in sched.columns:
            sched[col] = None

    # 3) Attach team ratings to schedule
    #    Left joins preserve all games even if a team rating is missing; we backfill with 50
    sched = sched.merge(
        teams[["team", "rating_0_100"]].rename(columns={"team": "home_team", "rating_0_100": "home_rating"}),
        on="home_team",
        how="left",
    ).merge(
        teams[["team", "rating_0_100"]].rename(columns={"team": "away_team", "rating_0_100": "away_rating"}),
        on="away_team",
        how="left",
    )

    sched["home_rating"] = sched["home_rating"].fillna(50.0)
    sched["away_rating"] = sched["away_rating"].fillna(50.0)

    # 4) Model home spread (internal home-positive)
    #    Neutral flag is integer 1/0 in our schedule builder
    sched["neutral_site"] = pd.to_numeric(sched["neutral_site"], errors="coerce").fillna(0).astype(int)
    sched["model_h"] = [
        _home_spread_from_ratings(h, a, n) for h, a, n in zip(sched["home_rating"], sched["away_rating"], sched["neutral_site"])
    ]

    # 5) Market: only apply for current_week; other weeks use 0 as requested
    sched["market_h"] = 0.0
    if market_df is not None and not market_df.empty and current_week is not None:
        m = market_df.copy()
        # Expecting book-style spreads in market_df['spread'] (negative = home favorite)
        # Convert to our internal home-positive first
        m["market_h_internal"] = -1.0 * pd.to_numeric(m["spread"], errors="coerce")
        # Only current week
        m = m.loc[pd.to_numeric(m["week"], errors="coerce").fillna(-1).astype(int) == int(current_week)]
        sched = sched.merge(
            m[["game_id", "market_h_internal"]], on="game_id", how="left", validate="one_to_one"
        )
        sched["market_h"] = sched["market_h_internal"].fillna(0.0)
        sched.drop(columns=["market_h_internal"], inplace=True, errors="ignore")

    # 6) Expected = market for now (placeholder hook for smoothing if you need it)
    sched["expected_h"] = sched["market_h"]

    # 7) Edge & Value
    #    Edge = model_h - market_h  (internal: home-positive). If book-style is requested, we’ll convert later.
    sched["edge_h"] = sched["model_h"] - sched["market_h"]

    #    Value heuristic: favor same-side agreement (model and expected push same way)
    same_side = np.sign(sched["model_h"]) == np.sign(sched["expected_h"])
    sched["value_h"] = np.where(same_side, np.abs(sched["edge_h"]), 0.0)

    #    Qualified bets: current week only, with magnitude threshold
    abs_edge = np.abs(sched["edge_h"])
    sched["qualified"] = np.where(
        (current_week is not None)
        & (pd.to_numeric(sched["week"], errors="coerce").astype("Int64") == int(current_week))
        & (abs_edge >= 2.0),
        "✓",
        "—",
    )

    # 8) Convert to book-style (negative = home favorite) for UI if requested
    if force_book_style:
        sched["MODEL (H)"] = _book_style(sched["model_h"]).round(1)
        sched["MARKET (H)"] = _book_style(sched["market_h"]).round(1)
        sched["EXPECTED (H)"] = _book_style(sched["expected_h"]).round(1)
        # Edge shown in UI as absolute magnitude with sign meaning "book-style model - market"
        # Converting internal edge to book-style: -(model_h - market_h) == (book_model - book_market)
        sched["EDGE"] = (-1.0 * sched["edge_h"]).round(1)
        # Keep value as a positive score
        sched["VALUE"] = sched["value_h"].round(1)
    else:
        # Raw internal (home-positive) naming
        sched["MODEL (H)"] = sched["model_h"].round(1)
        sched["MARKET (H)"] = sched["market_h"].round(1)
        sched["EXPECTED (H)"] = sched["expected_h"].round(1)
        sched["EDGE"] = sched["edge_h"].round(1)
        sched["VALUE"] = sched["value_h"].round(1)

    # 9) Final tidy
    out = sched.rename(
        columns={
            "week": "WEEK",
            "date": "DATE",
            "away_team": "AWAY",
            "home_team": "HOME",
            "neutral_site": "NEUTRAL",
        }
    )

    # NEUTRAL indicator for UI
    out["NEUTRAL"] = np.where(out["NEUTRAL"].astype(int) == 1, "Y", "—")

    cols = [
        "WEEK",
        "DATE",
        "AWAY",
        "HOME",
        "NEUTRAL",
        "MODEL (H)",
        "MARKET (H)",
        "EXPECTED (H)",
        "EDGE",
        "VALUE",
        "qualified",
        "game_id",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan if c not in ("qualified",) else "—"

    # Sort stable
    out = out[cols].sort_values(["WEEK", "DATE", "AWAY", "HOME"], kind="stable").reset_index(drop=True)
    return out