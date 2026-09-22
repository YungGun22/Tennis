import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional
from database.db import DatabaseManager
from src.features.serve import compute_opponent_adjusted_serve
from src.features.return import compute_opponent_adjusted_return
from src.features.form import calculate_rolling_form
from src.features.fatigue import calculate_fatigue_and_workload

class FeaturePipeline:
    """Orchestrates chronological leak-free feature extraction across the database."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def build_all_features(self) -> None:
        conn = self.db.get_connection()
        
        # Reset table
        conn.execute("DELETE FROM rolling_features;")
        conn.commit()

        print("[FEATURES] Fetching completed historical match records...")
        matches_df = pd.read_sql_query(
            "SELECT match_id, tour_date, surface, winner_id, loser_id FROM matches WHERE is_completed = 1 ORDER BY tour_date ASC;",
            conn
        )

        stats_df = pd.read_sql_query(
            "SELECT * FROM player_match_stats JOIN matches USING(match_id);",
            conn
        )

        rolling_records = []
        print(f"[FEATURES] Computing rolling features for {len(matches_df)} matches...")

        for _, row in matches_df.iterrows():
            m_id = row["match_id"]
            m_date = datetime.strptime(row["tour_date"], "%Y-%m-%d")
            surf = row["surface"]
            w_id = row["winner_id"]
            l_id = row["loser_id"]

            # Filter historical records prior to current match date (Strict Leakage Prevention)
            prior_stats = stats_df[stats_df["tour_date"] < row["tour_date"]]
            prior_matches = matches_df[matches_df["tour_date"] < row["tour_date"]]

            for p_id in [w_id, l_id]:
                p_stats = prior_stats[prior_stats["player_id"] == p_id]
                
                # Match history format
                p_m_w = prior_matches[prior_matches["winner_id"] == p_id].copy()
                p_m_w["is_winner"] = 1
                p_m_l = prior_matches[prior_matches["loser_id"] == p_id].copy()
                p_m_l["is_winner"] = 0
                p_history = pd.concat([p_m_w, p_m_l]).sort_values("tour_date")

                # Serve/Return strengths
                adj_serve = compute_opponent_adjusted_serve(p_stats, pd.DataFrame())
                adj_return = compute_opponent_adjusted_return(p_stats, pd.DataFrame())

                # Form & Fatigue
                form_dict = calculate_rolling_form(p_history, surf)
                fatigue_dict = calculate_fatigue_and_workload(p_history, m_date)

                rolling_records.append({
                    "match_id": m_id,
                    "player_id": p_id,
                    "win_pct_last_5": form_dict["win_pct_last_5"],
                    "win_pct_last_10": form_dict["win_pct_last_10"],
                    "win_pct_last_20": form_dict["win_pct_last_20"],
                    "surface_win_pct_last_10": form_dict["surface_win_pct_last_10"],
                    "adj_serve_strength": adj_serve,
                    "adj_return_strength": adj_return,
                    "rest_days": fatigue_dict["rest_days"],
                    "matches_last_7_days": fatigue_dict["matches_last_7_days"],
                    "matches_last_14_days": fatigue_dict["matches_last_14_days"],
                    "sets_last_14_days": fatigue_dict["sets_last_14_days"],
                    "minutes_last_14_days": fatigue_dict["minutes_last_14_days"]
                })

        print(f"[FEATURES] Saving {len(rolling_records)} feature snapshots to SQLite...")
        df_rolling = pd.DataFrame(rolling_records)
        df_rolling.to_sql("rolling_features", conn, if_exists="append", index=False, chunksize=5000)
        conn.close()
        print("[SUCCESS] Feature engineering process complete.")

if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.build_all_features()