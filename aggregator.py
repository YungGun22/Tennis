from typing import Dict, Any
import numpy as np

def generate_prediction_summary(
    player_a_name: str,
    player_b_name: str,
    simulation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Formats raw simulation output into structured match prediction objects."""
    probs = simulation_results["score_distribution"]
    
    # Sort score outcomes by probability
    sorted_scores = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    most_likely_score, highest_prob = sorted_scores[0]

    return {
        "matchup": f"{player_a_name} vs {player_b_name}",
        "predicted_winner": player_a_name if simulation_results["win_probability_a"] >= 0.5 else player_b_name,
        "p1_win_probability": round(simulation_results["win_probability_a"], 4),
        "p2_win_probability": round(simulation_results["win_probability_b"], 4),
        "most_likely_exact_score": most_likely_score,
        "exact_score_confidence": round(highest_prob, 4),
        "exact_score_probabilities": {k: round(v, 4) for k, v in probs.items()},
        "expected_total_sets": round(simulation_results["avg_sets_played"], 2)
    }