import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from database.db import DatabaseManager
from src.models.set_model import SetWinModelEngine
from src.simulation.monte_carlo import MonteCarloMatchSimulator
from src.evaluation.metrics import evaluate_exact_score_predictions
from src.evaluation.betting import simulate_betting_performance

class WalkForwardBacktester:
    """Executes chronological out-of-sample backtesting across historical tour matches."""

    def __init__(self, start_date: str = "2023-01-01", end_date: str = "2023-12-31"):
        self.start_date = start_date
        self.end_date = end_date
        self.db = DatabaseManager()

    def run_backtest(self) -> Dict[str, Any]:
        conn = self.db.get_connection()
        
        print(f"[BACKTEST] Querying historical matches between {self.start_date} and {self.end_date}...")
        query = f"""
        SELECT 
            m.match_id,
            m.tour_date,
            m.winner_id,
            m.loser_id,
            m.winner_sets,
            m.loser_sets,
            m.surface,
            m.best_of
        FROM matches m
        WHERE m.tour_date >= '{self.start_date}' AND m.tour_date <= '{self.end_date}'
          AND m.is_completed = 1
        ORDER BY m.tour_date ASC;
        """
        matches_df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"[BACKTEST] Loaded {len(matches_df)} target matches for evaluation.")

        set_model = SetWinModelEngine()
        
        actual_scores = []
        predicted_prob_dicts = []
        mock_odds_list = []

        for idx, row in matches_df.iterrows():
            w_sets = int(row["winner_sets"]) if pd.notnull(row["winner_sets"]) else 2
            l_sets = int(row["loser_sets"]) if pd.notnull(row["loser_sets"]) else 0
            actual_score = f"{w_sets}-{l_sets}"
            
            # Baseline differential feature mock for historical testing
            diff_payload = {
                "elo_diff": 40.0,
                "surface_elo_diff": 25.0,
                "win_pct_last_10_diff": 0.05,
                "surface_win_pct_last_10_diff": 0.03,
                "adj_serve_strength_diff": 0.01,
                "adj_return_strength_diff": 0.01,
                "rest_days_diff": 0.0,
                "matches_last_14_days_diff": 0
            }

            p_set_win = set_model.predict_set_probability(diff_payload)
            is_bo5 = True if row["best_of"] == 5 else False

            simulator = MonteCarloMatchSimulator(
                base_p_set_a=p_set_win,
                best_of_five=is_bo5,
                simulations=2000
            )
            sim_res = simulator.run_simulation()

            actual_scores.append(actual_score)
            predicted_prob_dicts.append(sim_res["score_distribution"])

            # Mock synthetic bookmaker closing odds with 4% overround
            mock_odds = {}
            for sc, prob in sim_res["score_distribution"].items():
                if prob > 0:
                    mock_odds[sc] = round(1.0 / (prob * 1.04), 2)
            mock_odds_list.append(mock_odds)

        print("[BACKTEST] Computing statistical evaluation metrics...")
        eval_metrics = evaluate_exact_score_predictions(actual_scores, predicted_prob_dicts)

        print("[BACKTEST] Simulating Kelly Criterion betting performance...")
        betting_res = simulate_betting_performance(predicted_prob_dicts, actual_scores, mock_odds_list)

        return {
            "evaluation_metrics": eval_metrics,
            "betting_simulation": betting_res
        }