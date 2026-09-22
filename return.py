import pandas as pd
import numpy as np

def compute_opponent_adjusted_return(
    player_stats_history: pd.DataFrame,
    opponent_serve_stats_history: pd.DataFrame,
    default_break_rate: float = 0.20
) -> float:
    """Calculates opponent-adjusted return strength index (RSI).
    
    RSI = (Player's Historical Break Rate) - (Avg Opponent's Historical Hold Loss Rate)
    """
    if player_stats_history.empty:
        return 0.0

    bp_opps = player_stats_history["break_points_opportunities"].sum()
    bp_converted = player_stats_history["break_points_converted"].sum()

    if bp_opps == 0 or pd.isna(bp_opps):
        return 0.0

    raw_break_rate = bp_converted / bp_opps

    return raw_break_rate - default_break_rate