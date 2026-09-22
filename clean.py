import re
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from database.db import DatabaseManager

class DataCleaner:
    """Executes data hygiene operations across raw match logs and player profiles."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    @staticmethod
    def normalize_player_name(name: str) -> str:
        """Converts name variants (e.g., 'Zverev, Alexander', 'Alexander Zverev ') into 'Alexander Zverev'."""
        if not name or pd.isna(name):
            return "Unknown Player"
        
        name = str(name).strip()
        
        # Handle "Lastname, Firstname" format
        if "," in name:
            parts = name.split(",", 1)
            name = f"{parts[1].strip()} {parts[0].strip()}"
            
        # Normalize double spaces and standard capitalization
        name = re.sub(r"\s+", " ", name)
        return name.title()

    @staticmethod
    def parse_score_string(score: str) -> Dict[str, Union[int, bool, str]]:
        """Parses raw tennis scores into structured set counts and match completion indicators.
        
        Example:
            '6-4 3-6 7-6(5)' -> winner_sets: 2, loser_sets: 1, completed: True
            '6-2 3-0 RET'    -> winner_sets: 1, loser_sets: 0, completed: False (Retirement)
        """
        if not score or pd.isna(score) or str(score).strip() == "":
            return {"winner_sets": 0, "loser_sets": 0, "total_games": 0, "completed": False, "status": "UNKNOWN"}

        score_clean = str(score).strip().upper()
        
        # Detect incomplete matches
        is_completed = True
        status = "COMPLETED"
        if any(term in score_clean for term in ["RET", "W/O", "DEF", "ABD"]):
            is_completed = False
            if "RET" in score_clean:
                status = "RETIRED"
            elif "W/O" in score_clean:
                status = "WALKOVER"
            else:
                status = "DEFAULT"

        # Clean score tokens (remove tiebreak details in parentheses and retirement markers)
        score_tokens = re.sub(r"\([0-9]+\)", "", score_clean)
        score_tokens = re.sub(r"[A-Z/]+", "", score_tokens).strip()

        winner_sets = 0
        loser_sets = 0
        total_games = 0

        sets = score_tokens.split()
        for set_score in sets:
            if "-" not in set_score:
                continue
            parts = set_score.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                w_games = int(parts[0])
                l_games = int(parts[1])
                total_games += (w_games + l_games)
                
                if w_games > l_games:
                    winner_sets += 1
                elif l_games > w_games:
                    loser_sets += 1

        return {
            "winner_sets": winner_sets,
            "loser_sets": loser_sets,
            "total_games": total_games,
            "completed": is_completed,
            "status": status
        }

    def clean_database() -> None:
        """Executes full SQL database cleaning pass."""
        conn = self.db.get_connection()
        
        print("[CLEAN] 1. Standardizing player names...")
        players_df = pd.read_sql_query("SELECT player_id, canonical_name FROM players", conn)
        players_df["canonical_name_clean"] = players_df["canonical_name"].apply(self.normalize_player_name)
        
        # Batch update players table
        update_players_sql = "UPDATE players SET canonical_name = ? WHERE player_id = ?"
        self.db.execute_batch(update_players_sql, list(zip(players_df["canonical_name_clean"], players_df["player_id"])))

        print("[CLEAN] 2. Cleaning match scores and detecting incomplete matches...")
        matches_df = pd.read_sql_query("SELECT match_id, score_string FROM matches", conn)
        
        matches_clean = []
        for _, row in matches_df.iterrows():
            parsed = self.parse_score_string(row["score_string"])
            matches_clean.append((
                parsed["completed"],
                parsed["winner_sets"],
                parsed["loser_sets"],
                parsed["total_games"],
                row["match_id"]
            ))

        # Add tracking columns if missing
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE matches ADD COLUMN is_completed INTEGER DEFAULT 1;")
            cursor.execute("ALTER TABLE matches ADD COLUMN winner_sets INTEGER;")
            cursor.execute("ALTER TABLE matches ADD COLUMN loser_sets INTEGER;")
            cursor.execute("ALTER TABLE matches ADD COLUMN total_games INTEGER;")
            conn.commit()
        except Exception:
            pass  # Columns already exist

        update_matches_sql = """
            UPDATE matches 
            SET is_completed = ?, winner_sets = ?, loser_sets = ?, total_games = ? 
            WHERE match_id = ?
        """
        self.db.execute_batch(update_matches_sql, matches_clean)

        print("[CLEAN] 3. Filtering extreme ranking anomalies...")
        # Handle unranked/anomalous ranks (e.g., > 2000 set to null for smooth Elo/Ranking interpolation)
        conn.execute("UPDATE matches SET winner_rank = NULL WHERE winner_rank > 2500 OR winner_rank <= 0;")
        conn.execute("UPDATE matches SET loser_rank = NULL WHERE loser_rank > 2500 OR loser_rank <= 0;")
        conn.commit()
        
        conn.close()
        print("[SUCCESS] Data cleaning phase complete.")

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.clean_database()