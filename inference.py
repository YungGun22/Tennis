import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from database.db import DatabaseManager
from src.models.set_model import SetWinModelEngine
from src.simulation.monte_carlo import MonteCarloMatchSimulator

class RealTimeInferenceEngine:
    """Orchestrates live feature extraction, ML prediction, and simulation for new match queries."""

    def __init__(self):
        self.db = DatabaseManager()
        self.set_model = SetWinModelEngine()

    def _fetch_player_latest_features(self, player_id: int, surface: str, conn) -> Dict[str, float]:
        """Queries the database for a player's most recent rolling features and Elo ratings."""
        # Fetch latest rolling features
        query_rf = f"""
        SELECT win_pct_last_10, surface_win_pct_last_10, adj_serve_strength, 
               adj_return_strength, rest_days, matches_last_14_days
        FROM rolling_features
        WHERE player_id = {player_id}
        ORDER BY rowid DESC LIMIT 1;
        """
        rf_df = pd.read_sql_query(query_rf, conn)

        # Fetch latest Elo ratings
        query_elo = f"""
        SELECT elo_rating, surface_elo_rating
        FROM elo_ratings
        WHERE player_id = {player_id} AND surface = '{surface}'
        ORDER BY tour_date DESC LIMIT 1;
        """
        elo_df = pd.read_sql_query(query_elo, conn)

        defaults = {
            "win_pct_last_10": 0.50,
            "surface_win_pct_last_10": 0.50,
            "adj_serve_strength": 0.0,
            "adj_return_strength": 0.0,
            "rest_days": 7.0,
            "matches_last_14_days": 0,
            "elo_rating": 1500.0,
            "surface_elo_rating": 1500.0
        }

        if not rf_df.empty:
            defaults.update(rf_df.iloc[0].to_dict())
        if not elo_df.empty:
            defaults.update(elo_df.iloc[0].to_dict())

        return defaults

    def predict_match(
        self,
        player_a_id: int,
        player_b_id: int,
        surface: str,
        best_of: int = 3,
        simulations: int = 10000
    ) -> Dict[str, Any]:
        conn = self.db.get_connection()

        p_a = self._fetch_player_latest_features(player_a_id, surface, conn)
        p_b = self._fetch_player_latest_features(player_b_id, surface, conn)
        conn.close()

        # Construct differential payload
        diff_payload = {
            "elo_diff": p_a["elo_rating"] - p_b["elo_rating"],
            "surface_elo_diff": p_a["surface_elo_rating"] - p_b["surface_elo_rating"],
            "win_pct_last_10_diff": p_a["win_pct_last_10"] - p_b["win_pct_last_10"],
            "surface_win_pct_last_10_diff": p_a["surface_win_pct_last_10"] - p_b["surface_win_pct_last_10"],
            "adj_serve_strength_diff": p_a["adj_serve_strength"] - p_b["adj_serve_strength"],
            "adj_return_strength_diff": p_a["adj_return_strength"] - p_b["adj_return_strength"],
            "rest_days_diff": p_a["rest_days"] - p_b["rest_days"],
            "matches_last_14_days_diff": p_a["matches_last_14_days"] - p_b["matches_last_14_days"]
        }

        # Predict single set win probability
        base_p_set_a = self.set_model.predict_set_probability(diff_payload)

        # Run Monte Carlo simulation
        is_bo5 = True if best_of == 5 else False
        simulator = MonteCarloMatchSimulator(
            base_p_set_a=base_p_set_a,
            best_of_five=is_bo5,
            simulations=simulations
        )
        sim_res = simulator.run_simulation()

        # Extract top score
        score_probs = sim_res["score_distribution"]
        sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
        top_score, top_conf = sorted_scores[0]

        return {
            "status": "success",
            "player_a_id": player_a_id,
            "player_b_id": player_b_id,
            "player_a_win_probability": round(sim_res["win_probability_a"], 4),
            "player_b_win_probability": round(sim_res["win_probability_b"], 4),
            "single_set_win_prob_a": round(base_p_set_a, 4),
            "exact_scores": {
                "score_probabilities": {k: round(v, 4) for k, v in score_probs.items()},
                "most_likely_score": top_score,
                "confidence": round(top_conf, 4)
            },
            "expected_total_sets": round(sim_res["avg_sets_played"], 2)
        }