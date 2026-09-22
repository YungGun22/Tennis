import pandas as pd
import numpy as np

def compute_opponent_adjusted_serve(
    player_stats_history: pd.DataFrame,
    opponent_return_stats_history: pd.DataFrame,
    default_hold_rate: float = 0.80
) -> float:
    """Calculates opponent-adjusted serve strength index (SSI).
    
    SSI = (Player's Historical Hold Rate) - (Avg Opponent's Historical Break Rate against rest of tour)
    """
    if player_stats_history.empty:
        return 0.0

    total_service_games = player_stats_history["service_games_played"].sum()
    if total_service_games == 0 or pd.isna(total_service_games):
        return 0.0

    bp_faced = player_stats_history["break_points_faced"].sum()
    bp_saved = player_stats_history["break_points_saved"].sum()
    breaks_conceded = bp_faced - bp_saved
    holds = total_service_games - breaks_conceded

    raw_hold_rate = holds / total_service_games

    # Adjust relative to baseline
    avg_opp_break_rate = 0.20
    if not opponent_return_stats_history.empty:
        opp_bp_opportunities = opponent_return_stats_history["break_points_opportunities"].sum()
        opp_bp_converted = opponent_return_stats_history["break_points_converted"].sum()
        if opp_bp_opportunities > 0:
            avg_opp_break_rate = opp_bp_converted / opp_bp_opportunities

    return raw_hold_rate - (1.0 - avg_opp_break_rate)