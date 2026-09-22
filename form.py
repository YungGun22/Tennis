import pandas as pd
import numpy as np
from typing import Dict

def calculate_rolling_form(
    player_match_history: pd.DataFrame,
    current_surface: str,
    windows: list = [5, 10, 20]
) -> Dict[str, float]:
    """Calculates chronological rolling win rates and exponential weighted form."""
    form_metrics = {}

    if player_match_history.empty:
        for w in windows:
            form_metrics[f"win_pct_last_{w}"] = 0.50
        form_metrics["surface_win_pct_last_10"] = 0.50
        form_metrics["ewma_form"] = 0.50
        return form_metrics

    wins = player_match_history["is_winner"].values

    # Simple Rolling Averages
    for w in windows:
        recent = wins[-w:] if len(wins) >= w else wins
        form_metrics[f"win_pct_last_{w}"] = float(np.mean(recent)) if len(recent) > 0 else 0.50

    # Surface-Specific Rolling Form
    surface_matches = player_match_history[player_match_history["surface"] == current_surface]
    if not surface_matches.empty:
        recent_surf_wins = surface_matches["is_winner"].values[-10:]
        form_metrics["surface_win_pct_last_10"] = float(np.mean(recent_surf_wins))
    else:
        form_metrics["surface_win_pct_last_10"] = 0.50

    # Exponentially Weighted Moving Average Form
    if len(wins) > 0:
        s = pd.Series(wins)
        ewma = s.ewm(span=10, adjust=False).mean().iloc[-1]
        form_metrics["ewma_form"] = float(ewma)
    else:
        form_metrics["ewma_form"] = 0.50

    return form_metrics