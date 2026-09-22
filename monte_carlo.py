import numpy as np
import pandas as pd
from typing import Dict, Any

class MonteCarloMatchSimulator:
    """Stochastic match simulator modeling dynamic set transition probabilities."""

    def __init__(
        self,
        base_p_set_a: float,
        best_of_five: bool = False,
        momentum_boost: float = 0.03,
        fatigue_decay_a: float = 0.0,
        fatigue_decay_b: float = 0.0,
        simulations: int = 10000
    ):
        self.base_p_set_a = np.clip(base_p_set_a, 0.01, 0.99)
        self.best_of_five = best_of_five
        self.sets_to_win = 3 if best_of_five else 2
        self.momentum_boost = momentum_boost
        self.fatigue_decay_a = fatigue_decay_a
        self.fatigue_decay_b = fatigue_decay_b
        self.simulations = simulations

    def simulate_single_match(self) -> Dict[str, Any]:
        """Simulates one match with dynamic momentum and fatigue adjustments."""
        sets_a = 0
        sets_b = 0
        set_history = []

        current_p_a = self.base_p_set_a

        while sets_a < self.sets_to_win and sets_b < self.sets_to_win:
            set_number = len(set_history) + 1

            # Apply fatigue decay in deep sets (Set 4+)
            p_adj = current_p_a
            if set_number >= 4:
                p_adj += (self.fatigue_decay_b - self.fatigue_decay_a)

            # Apply momentum boost from previous set winner
            if set_history:
                last_winner = set_history[-1]
                if last_winner == "A":
                    p_adj += self.momentum_boost
                else:
                    p_adj -= self.momentum_boost

            p_adj = np.clip(p_adj, 0.01, 0.99)

            # Draw set winner
            if np.random.rand() < p_adj:
                sets_a += 1
                set_history.append("A")
            else:
                sets_b += 1
                set_history.append("B")

        score_str = f"{sets_a}-{sets_b}"
        return {"winner": "A" if sets_a > sets_b else "B", "score": score_str, "total_sets": len(set_history)}

    def run_simulation(self) -> Dict[str, Any]:
        """Runs batch stochastic simulations and returns exact score distributions."""
        score_counts = {}
        winner_counts = {"A": 0, "B": 0}
        total_sets_list = []

        for _ in range(self.simulations):
            res = self.simulate_single_match()
            sc = res["score"]
            w = res["winner"]
            
            score_counts[sc] = score_counts.get(sc, 0) + 1
            winner_counts[w] += 1
            total_sets_list.append(res["total_sets"])

        # Normalize to probability distributions
        score_probs = {k: v / self.simulations for k, v in score_counts.items()}
        
        return {
            "win_probability_a": winner_counts["A"] / self.simulations,
            "win_probability_b": winner_counts["B"] / self.simulations,
            "score_distribution": score_probs,
            "avg_sets_played": float(np.mean(total_sets_list))
        }