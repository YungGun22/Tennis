import pandas as pd
from typing import Dict

def calculate_shrunken_h2h(
    h2h_matches: pd.DataFrame,
    player_a_id: int,
    prior_prob_a: float = 0.50,
    m_weight: float = 5.0
) -> Dict[str, float]:
    """Calculates Bayesian shrinkage H2H win rate.
    
    Formula: P_shrunken = (n_wins_A + m * P_prior) / (n_total_meetings + m)
    """
    if h2h_matches.empty:
        return {
            "h2h_meetings": 0,
            "h2h_shrunken_win_pct_a": prior_prob_a
        }

    total_meetings = len(h2h_matches)
    a_wins = len(h2h_matches[h2h_matches["winner_id"] == player_a_id])

    shrunken_prob = (a_wins + m_weight * prior_prob_a) / (total_meetings + m_weight)

    return {
        "h2h_meetings": total_meetings,
        "h2h_shrunken_win_pct_a": float(shrunken_prob)
    }