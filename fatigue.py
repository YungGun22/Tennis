import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

def calculate_fatigue_and_workload(
    player_match_history: pd.DataFrame,
    current_match_date: datetime
) -> Dict[str, float]:
    """Computes days rest and match/set intensity over 7 and 14 day rolling windows."""
    if player_match_history.empty:
        return {
            "rest_days": 7.0,
            "matches_last_7_days": 0,
            "matches_last_14_days": 0,
            "sets_last_14_days": 0,
            "minutes_last_14_days": 0
        }

    history = player_match_history.copy()
    history["tour_date"] = pd.to_datetime(history["tour_date"])

    # Rest days
    last_match_date = history["tour_date"].max()
    rest_days = max(0.0, float((current_match_date - last_match_date).days))

    # Rolling window filters
    date_7d = current_match_date - timedelta(days=7)
    date_14d = current_match_date - timedelta(days=14)

    m_7d = history[history["tour_date"] >= date_7d]
    m_14d = history[history["tour_date"] >= date_14d]

    matches_7d = len(m_7d)
    matches_14d = len(m_14d)

    # Sets and minutes played
    sets_14d = int((m_14d["winner_sets"].fillna(2) + m_14d["loser_sets"].fillna(0)).sum())
    minutes_14d = float(m_14d["match_minutes"].fillna(90).sum())

    return {
        "rest_days": min(rest_days, 30.0), # cap ceiling
        "matches_last_7_days": matches_7d,
        "matches_last_14_days": matches_14d,
        "sets_last_14_days": sets_14d,
        "minutes_last_14_days": minutes_14d
    }