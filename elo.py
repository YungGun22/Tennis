import math
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from config import DB_PATH, ELO_BASE, ELO_DEFAULT_K, SURFACE_TYPES
from database.db import DatabaseManager

class SurfaceEloEngine:
    """Sequential, leak-free surface-aware Elo rating engine."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        base_elo: float = ELO_BASE,
        base_k: float = ELO_DEFAULT_K,
        surface_weight: float = 0.50,
        inactivity_decay_days: int = 120
    ):
        self.db = db_manager or DatabaseManager()
        self.base_elo = base_elo
        self.base_k = base_k
        self.surface_weight = surface_weight
        self.inactivity_decay_days = inactivity_decay_days

        # State stores: player_id -> current rating/match count
        self.overall_elo: Dict[int, float] = {}
        self.surface_elo: Dict[int, Dict[str, float]] = {}
        self.match_counts: Dict[int, int] = {}
        self.surface_match_counts: Dict[int, Dict[str, int]] = {}
        self.last_played_date: Dict[int, datetime] = {}

    def _get_player_overall_elo(self, player_id: int, current_date: datetime) -> float:
        """Retrieves overall Elo, applying inactivity regression toward baseline if applicable."""
        if player_id not in self.overall_elo:
            self.overall_elo[player_id] = self.base_elo
            self.match_counts[player_id] = 0
            return self.base_elo

        # Apply decay for prolonged inactivity
        if player_id in self.last_played_date:
            days_inactive = (current_date - self.last_played_date[player_id]).days
            if days_inactive > self.inactivity_decay_days:
                decay_periods = (days_inactive - self.inactivity_decay_days) // 30
                decay_factor = 0.98 ** decay_periods
                current_val = self.overall_elo[player_id]
                self.overall_elo[player_id] = self.base_elo + (current_val - self.base_elo) * decay_factor

        return self.overall_elo[player_id]

    def _get_player_surface_elo(self, player_id: int, surface: str, current_date: datetime) -> float:
        """Retrieves surface-specific Elo."""
        if player_id not in self.surface_elo:
            self.surface_elo[player_id] = {s: self.base_elo for s in SURFACE_TYPES}
            self.surface_match_counts[player_id] = {s: 0 for s in SURFACE_TYPES}

        if surface not in self.surface_elo[player_id]:
            self.surface_elo[player_id][surface] = self.base_elo
            self.surface_match_counts[player_id][surface] = 0

        return self.surface_elo[player_id][surface]

    def get_composite_elo(self, player_id: int, surface: str, current_date: datetime) -> Tuple[float, float, float]:
        """Calculates composite rating weighted between overall and surface ratings.
        
        Returns:
            (composite_elo, overall_elo, surface_elo)
        """
        ovr = self._get_player_overall_elo(player_id, current_date)
        surf = self._get_player_surface_elo(player_id, surface, current_date)
        
        composite = (1.0 - self.surface_weight) * ovr + self.surface_weight * surf
        return composite, ovr, surf

    @staticmethod
    def calculate_expected_probability(rating_a: float, rating_b: float) -> float:
        """Standard logistic Elo probability formula."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _compute_dynamic_k(self, match_count: int, tour_level: str) -> float:
        """Dynamic K-Factor scaling based on experience and tournament significance."""
        # Experience multiplier
        if match_count < 10:
            k_exp = self.base_k * 1.5
        elif match_count < 30:
            k_exp = self.base_k * 1.2
        else:
            k_exp = self.base_k

        # Tournament level multiplier
        level_mult = 1.0
        if tour_level in ["G", "Grand Slam"]:
            level_mult = 1.20
        elif tour_level in ["M", "Masters 1000"]:
            level_mult = 1.10

        return k_exp * level_mult

    @staticmethod
    def _calculate_margin_multiplier(winner_sets: int, loser_sets: int) -> float:
        """Margin of victory scaling (straight-sets wins boost rating adjustments)."""
        if winner_sets == 2 and loser_sets == 0:
            return 1.20  # Dominant 2-0 win
        elif winner_sets == 3 and loser_sets == 0:
            return 1.30  # Dominant 3-0 win
        elif winner_sets == 2 and loser_sets == 1:
            return 0.90  # Contested 2-1 win
        return 1.00

    def process_match_sequentially(
        self,
        match_id: str,
        tour_date_str: str,
        surface: str,
        tour_level: str,
        winner_id: int,
        loser_id: int,
        winner_sets: int,
        loser_sets: int
    ) -> Dict:
        """Captures pre-match Elo snapshots, processes outcome, updates states, and returns database record."""
        current_date = datetime.strptime(tour_date_str, "%Y-%m-%d") if tour_date_str else datetime.now()
        surface = surface.capitalize() if surface else "Hard"
        if surface not in SURFACE_TYPES:
            surface = "Hard"

        # 1. RECORD PRE-MATCH RATINGS (LEAK-FREE SNAPSHOT)
        w_comp_pre, w_ovr_pre, w_surf_pre = self.get_composite_elo(winner_id, surface, current_date)
        l_comp_pre, l_ovr_pre, l_surf_pre = self.get_composite_elo(loser_id, surface, current_date)

        # Expected win probabilities
        p_winner_expected = self.calculate_expected_probability(w_comp_pre, l_comp_pre)

        # 2. CALCULATE K-FACTORS & MARGIN ADJ
        w_k = self._compute_dynamic_k(self.match_counts.get(winner_id, 0), tour_level)
        l_k = self._compute_dynamic_k(self.match_counts.get(loser_id, 0), tour_level)
        margin_mult = self._calculate_margin_multiplier(winner_sets, loser_sets)

        # 3. COMPUTE POST-MATCH UPDATES
        w_delta_ovr = w_k * margin_mult * (1.0 - p_winner_expected)
        l_delta_ovr = l_k * margin_mult * (0.0 - (1.0 - p_winner_expected))

        w_ovr_post = w_ovr_pre + w_delta_ovr
        l_ovr_post = l_ovr_pre + l_delta_ovr

        w_surf_post = w_surf_pre + w_delta_ovr
        l_surf_post = l_surf_pre + l_delta_ovr

        # 4. UPDATE INTERNAL STATES FOR FUTURE MATCHES
        self.overall_elo[winner_id] = w_ovr_post
        self.overall_elo[loser_id] = l_ovr_post
        self.surface_elo[winner_id][surface] = w_surf_post
        self.surface_elo[loser_id][surface] = l_surf_post

        self.match_counts[winner_id] = self.match_counts.get(winner_id, 0) + 1
        self.match_counts[loser_id] = self.match_counts.get(loser_id, 0) + 1
        self.surface_match_counts[winner_id][surface] = self.surface_match_counts[winner_id].get(surface, 0) + 1
        self.surface_match_counts[loser_id][surface] = self.surface_match_counts[loser_id].get(surface, 0) + 1

        self.last_played_date[winner_id] = current_date
        self.last_played_date[loser_id] = current_date

        return {
            "winner_record": {
                "match_id": match_id,
                "player_id": winner_id,
                "pre_match_overall_elo": w_ovr_pre,
                "post_match_overall_elo": w_ovr_post,
                "pre_match_surface_elo": w_surf_pre,
                "post_match_surface_elo": w_surf_post,
                "surface": surface
            },
            "loser_record": {
                "match_id": match_id,
                "player_id": loser_id,
                "pre_match_overall_elo": l_ovr_pre,
                "post_match_overall_elo": l_ovr_post,
                "pre_match_surface_elo": l_surf_pre,
                "post_match_surface_elo": l_surf_post,
                "surface": surface
            }
        }

    def process_all_historical_matches(self) -> None:
        """Processes all clean historical matches chronologically and populates `elo_history`."""
        conn = self.db.get_connection()
        
        # Clear prior Elo records
        conn.execute("DELETE FROM elo_history;")
        conn.commit()

        print("[ELO] Fetching historical matches in strict chronological order...")
        query = """
            SELECT match_id, tour_date, surface, tournament_level, winner_id, loser_id, 
                   COALESCE(winner_sets, 2) as winner_sets, COALESCE(loser_sets, 0) as loser_sets
            FROM matches
            WHERE is_completed = 1
            ORDER BY tour_date ASC, match_id ASC;
        """
        df_matches = pd.read_sql_query(query, conn)
        print(f"[ELO] Processing {len(df_matches)} completed matches...")

        elo_records = []
        for _, row in df_matches.iterrows():
            rec = self.process_match_sequentially(
                match_id=row["match_id"],
                tour_date_str=row["tour_date"],
                surface=row["surface"],
                tour_level=row["tournament_level"],
                winner_id=row["winner_id"],
                loser_id=row["loser_id"],
                winner_sets=row["winner_sets"],
                loser_sets=row["loser_sets"]
            )
            elo_records.append(rec["winner_record"])
            elo_records.append(rec["loser_record"])

        # Batch insert pre-match Elo snapshots
        print(f"[ELO] Persisting {len(elo_records)} pre-match Elo snapshots to database...")
        df_elo = pd.DataFrame(elo_records)
        df_elo.to_sql("elo_history", conn, if_exists="append", index=False, chunksize=5000)
        conn.close()
        print("[SUCCESS] Surface Elo computation complete.")

if __name__ == "__main__":
    engine = SurfaceEloEngine()
    engine.process_all_historical_matches()