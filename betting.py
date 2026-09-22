import numpy as np
from typing import Dict, List, Any

def calculate_kelly_stake(
    predicted_prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05
) -> float:
    """Calculates Fractional Kelly Criterion stake size.
    
    f* = (p * b - q) / b, where b = decimal_odds - 1, q = 1 - p
    """
    if decimal_odds <= 1.0 or predicted_prob <= 0.0:
        return 0.0

    b = decimal_odds - 1.0
    q = 1.0 - predicted_prob
    kelly_f = (predicted_prob * b - q) / b

    if kelly_f <= 0.0:
        return 0.0

    # Apply fractional sizing and hard cap
    stake_fraction = min(kelly_f * fraction, max_stake_pct)
    return float(stake_fraction)

def simulate_betting_performance(
    predictions: List[Dict[str, float]],
    actual_scores: List[str],
    closing_odds_list: List[Dict[str, float]],
    initial_bankroll: float = 10000.0,
    min_ev_threshold: float = 0.03
) -> Dict[str, Any]:
    """Simulates betting portfolio performance against closing odds."""
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bets_placed = 0
    bets_won = 0

    for probs, actual, odds in zip(predictions, actual_scores, closing_odds_list):
        for score, decimal_odd in odds.items():
            if decimal_odd <= 1.0:
                continue

            p = probs.get(score, 0.0)
            expected_value = (p * decimal_odd) - 1.0

            if expected_value >= min_ev_threshold:
                stake_pct = calculate_kelly_stake(p, decimal_odd)
                stake_amount = bankroll * stake_pct

                if stake_amount <= 0:
                    continue

                bets_placed += 1
                if score == actual:
                    profit = stake_amount * (decimal_odd - 1.0)
                    bankroll += profit
                    bets_won += 1
                else:
                    bankroll -= stake_amount

                bankroll_history.append(bankroll)

    total_return = (bankroll - initial_bankroll) / initial_bankroll
    win_rate = bets_won / bets_placed if bets_placed > 0 else 0.0

    return {
        "initial_bankroll": initial_bankroll,
        "final_bankroll": round(bankroll, 2),
        "total_roi_pct": round(total_return * 100, 2),
        "total_bets_placed": bets_placed,
        "bet_win_rate": round(win_rate, 4),
        "max_bankroll": round(float(np.max(bankroll_history)), 2),
        "min_bankroll": round(float(np.min(bankroll_history)), 2)
    }