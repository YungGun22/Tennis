import os
import sqlite3
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from config import RAW_DATA_DIR, DATA_SOURCES, START_YEAR, END_YEAR, DB_PATH
from database.db import DatabaseManager

class DataIngestionEngine:
    """Handles automatic fetching of historical raw data, schema parsing, and SQLite ingestion."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()

    def download_file(self, url: str, dest_path: Path) -> bool:
        """Helper to safely fetch a remote CSV resource."""
        try:
            print(f"[FETCHING] {url}")
            urllib.request.urlretrieve(url, dest_path)
            return True
        except Exception as e:
            print(f"[WARNING] Could not fetch {url}: {e}")
            if dest_path.exists():
                os.remove(dest_path)
            return False

    def fetch_raw_data(self, start_year: int = START_YEAR, end_year: int = END_YEAR) -> None:
        """Downloads players and annual match records for ATP and WTA datasets."""
        # 1. Fetch Players
        self.download_file(DATA_SOURCES["atp_players"], RAW_DATA_DIR / "atp_players.csv")
        self.download_file(DATA_SOURCES["wta_players"], RAW_DATA_DIR / "wta_players.csv")

        # 2. Fetch Annual Matches
        for year in range(start_year, end_year + 1):
            atp_url = DATA_SOURCES["atp_matches"].format(year=year)
            atp_dest = RAW_DATA_DIR / f"atp_matches_{year}.csv"
            self.download_file(atp_url, atp_dest)

            wta_url = DATA_SOURCES["wta_matches"].format(year=year)
            wta_dest = RAW_DATA_DIR / f"wta_matches_{year}.csv"
            self.download_file(wta_url, wta_dest)

    def ingest_players((self) -> None:
        """Parses and populates the `players` table."""
        players_dfs = []
        for file_name in ["atp_players.csv", "wta_players.csv"]:
            path = RAW_DATA_DIR / file_name
            if path.exists():
                df = pd.read_csv(
                    path,
                    header=None,
                    names=["player_id", "first_name", "last_name", "hand", "birth_date", "country_code", "height_cm"],
                    dtype=str
                )
                players_dfs.append(df)

        if not players_dfs:
            print("[ERROR] No player data found to ingest.")
            return

        df_all = pd.concat(players_dfs, ignore_index=True)
        df_all["canonical_name"] = (df_all["first_name"].fillna("") + " " + df_all["last_name"].fillna("")).str.strip()
        
        # Clean height and dates
        df_all["height_cm"] = pd.to_numeric(df_all["height_cm"], errors="coerce")
        df_all["player_id"] = pd.to_numeric(df_all["player_id"], errors="coerce")
        
        # Standardize Handedness
        df_all["hand"] = df_all["hand"].str.upper().fillna("U")
        df_all.loc[~df_all["hand"].isin(["R", "L"]), "hand"] = "U"

        # Format birth_date to YYYY-MM-DD
        df_all["birth_date"] = pd.to_datetime(df_all["birth_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")

        df_players = df_all[["player_id", "canonical_name", "hand", "birth_date", "country_code", "height_cm"]].drop_duplicates(subset=["player_id"])

        conn = self.db_manager.get_connection()
        df_players.to_sql("players", conn, if_exists="append", index=False, method="multi", chunksize=1000)
        conn.close()
        print(f"[SUCCESS] Ingested {len(df_players)} player records.")

    def ingest_matches(self) -> None:
        """Parses raw annual match CSV files and populates `matches` and `player_match_stats` tables."""
        match_files = list(RAW_DATA_DIR.glob("atp_matches_*.csv")) + list(RAW_DATA_DIR.glob("wta_matches_*.csv"))
        
        if not match_files:
            print("[ERROR] No match files found for ingestion.")
            return

        all_matches = []
        all_stats = []

        for filepath in sorted(match_files):
            try:
                df = pd.read_csv(filepath, low_memory=False)
            except Exception as e:
                print(f"[WARNING] Could not read {filepath}: {e}")
                continue

            if df.empty:
                continue

            # Compute standardized date
            df["tour_date"] = pd.to_datetime(df["tour_date"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
            
            # Surface fallback
            df["surface"] = df["surface"].fillna("Hard").str.capitalize()
            df.loc[~df["surface"].isin(["Hard", "Clay", "Grass", "Carpet"]), "surface"] = "Hard"

            for idx, row in df.iterrows():
                match_id = f"{row.get('tourney_id', 'UNKNOWN')}_{row.get('match_num', idx)}"
                
                winner_id = row.get("winner_id")
                loser_id = row.get("loser_id")
                
                if pd.isna(winner_id) or pd.isna(loser_id) or pd.isna(row["tour_date"]):
                    continue

                tour_year = int(row["tour_date"][:4]) if row["tour_date"] else 2000
                best_of = int(row.get("best_of", 3)) if not pd.isna(row.get("best_of")) else 3
                
                # Match Record
                match_entry = {
                    "match_id": match_id,
                    "tour_year": tour_year,
                    "tour_date": row["tour_date"],
                    "tournament_name": str(row.get("tourney_name", "Unknown")),
                    "tournament_level": str(row.get("tourney_level", "A")),
                    "surface": row["surface"],
                    "indoor_flag": 1 if "indoor" in str(row.get("tourney_name", "")).lower() else 0,
                    "best_of": best_of,
                    "round": str(row.get("round", "R32")),
                    "winner_id": int(winner_id),
                    "loser_id": int(loser_id),
                    "score_string": str(row.get("score", "")),
                    "winner_rank": pd.to_numeric(row.get("winner_rank"), errors="coerce"),
                    "winner_rank_points": pd.to_numeric(row.get("winner_rank_points"), errors="coerce"),
                    "loser_rank": pd.to_numeric(row.get("loser_rank"), errors="coerce"),
                    "loser_rank_points": pd.to_numeric(row.get("loser_rank_points"), errors="coerce"),
                    "match_minutes": pd.to_numeric(row.get("minutes"), errors="coerce")
                }
                all_matches.append(match_entry)

                # Extract Statistics for Winner and Loser
                w_svpt = pd.to_numeric(row.get("w_svpt"), errors="coerce")
                l_svpt = pd.to_numeric(row.get("l_svpt"), errors="coerce")
                
                # Winner Stats
                if not pd.isna(w_svpt) and w_svpt > 0:
                    w_1stin = pd.to_numeric(row.get("w_1stIn"), errors="coerce") or 0
                    w_1stwon = pd.to_numeric(row.get("w_1stWon"), errors="coerce") or 0
                    w_2ndwon = pd.to_numeric(row.get("w_2ndWon"), errors="coerce") or 0
                    l_1stin = pd.to_numeric(row.get("l_1stIn"), errors="coerce") or 0
                    l_1stwon = pd.to_numeric(row.get("l_1stWon"), errors="coerce") or 0
                    l_2ndwon = pd.to_numeric(row.get("l_2ndWon"), errors="coerce") or 0

                    # Derived return metrics (Opponent's serving statistics)
                    w_ret_played = l_svpt
                    w_ret_won = l_svpt - (l_1stwon + l_2ndwon) if not pd.isna(l_svpt) else np.nan
                    
                    all_stats.append({
                        "match_id": match_id,
                        "player_id": int(winner_id),
                        "opponent_id": int(loser_id),
                        "is_winner": 1,
                        "aces": pd.to_numeric(row.get("w_ace"), errors="coerce"),
                        "double_faults": pd.to_numeric(row.get("w_df"), errors="coerce"),
                        "service_points_played": w_svpt,
                        "first_serves_in": w_1stin,
                        "first_serve_points_won": w_1stwon,
                        "second_serve_points_won": w_2ndwon,
                        "service_games_played": pd.to_numeric(row.get("w_SvGms"), errors="coerce"),
                        "break_points_saved": pd.to_numeric(row.get("w_bpSaved"), errors="coerce"),
                        "break_points_faced": pd.to_numeric(row.get("w_bpFaced"), errors="coerce"),
                        "return_points_played": w_ret_played,
                        "return_points_won": w_ret_won,
                        "first_return_points_won": (l_1stin - l_1stwon) if l_1stin >= l_1stwon else 0,
                        "second_return_points_won": (l_svpt - l_1stin - l_2ndwon) if l_svpt >= (l_1stin + l_2ndwon) else 0,
                        "break_points_converted": (pd.to_numeric(row.get("l_bpFaced"), errors="coerce") or 0) - (pd.to_numeric(row.get("l_bpSaved"), errors="coerce") or 0),
                        "break_points_opportunities": pd.to_numeric(row.get("l_bpFaced"), errors="coerce"),
                        "return_games_played": pd.to_numeric(row.get("l_SvGms"), errors="coerce")
                    })

                # Loser Stats
                if not pd.isna(l_svpt) and l_svpt > 0:
                    w_1stin = pd.to_numeric(row.get("w_1stIn"), errors="coerce") or 0
                    w_1stwon = pd.to_numeric(row.get("w_1stWon"), errors="coerce") or 0
                    w_2ndwon = pd.to_numeric(row.get("w_2ndWon"), errors="coerce") or 0
                    l_1stin = pd.to_numeric(row.get("l_1stIn"), errors="coerce") or 0
                    l_1stwon = pd.to_numeric(row.get("l_1stWon"), errors="coerce") or 0
                    l_2ndwon = pd.to_numeric(row.get("l_2ndWon"), errors="coerce") or 0

                    l_ret_played = w_svpt
                    l_ret_won = w_svpt - (w_1stwon + w_2ndwon) if not pd.isna(w_svpt) else np.nan

                    all_stats.append({
                        "match_id": match_id,
                        "player_id": int(loser_id),
                        "opponent_id": int(winner_id),
                        "is_winner": 0,
                        "aces": pd.to_numeric(row.get("l_ace"), errors="coerce"),
                        "double_faults": pd.to_numeric(row.get("l_df"), errors="coerce"),
                        "service_points_played": l_svpt,
                        "first_serves_in": l_1stin,
                        "first_serve_points_won": l_1stwon,
                        "second_serve_points_won": l_2ndwon,
                        "service_games_played": pd.to_numeric(row.get("l_SvGms"), errors="coerce"),
                        "break_points_saved": pd.to_numeric(row.get("l_bpSaved"), errors="coerce"),
                        "break_points_faced": pd.to_numeric(row.get("l_bpFaced"), errors="coerce"),
                        "return_points_played": l_ret_played,
                        "return_points_won": l_ret_won,
                        "first_return_points_won": (w_1stin - w_1stwon) if w_1stin >= w_1stwon else 0,
                        "second_return_points_won": (w_svpt - w_1stin - w_2ndwon) if w_svpt >= (w_1stin + w_2ndwon) else 0,
                        "break_points_converted": (pd.to_numeric(row.get("w_bpFaced"), errors="coerce") or 0) - (pd.to_numeric(row.get("w_bpSaved"), errors="coerce") or 0),
                        "break_points_opportunities": pd.to_numeric(row.get("w_bpFaced"), errors="coerce"),
                        "return_games_played": pd.to_numeric(row.get("w_SvGms"), errors="coerce")
                    })

        # Save to SQLite in Chunk Batches
        conn = self.db_manager.get_connection()
        
        if all_matches:
            df_matches = pd.DataFrame(all_matches).drop_duplicates(subset=["match_id"])
            df_matches.to_sql("matches", conn, if_exists="append", index=False, chunksize=2000)
            print(f"[SUCCESS] Ingested {len(df_matches)} match records.")

        if all_stats:
            df_stats = pd.DataFrame(all_stats)
            df_stats.to_sql("player_match_stats", conn, if_exists="append", index=False, chunksize=5000)
            print(f"[SUCCESS] Ingested {len(df_stats)} player match performance records.")

        conn.close()

if __name__ == "__main__":
    db = DatabaseManager()
    db.initialize_schema()
    
    ingestor = DataIngestionEngine(db)
    ingestor.fetch_raw_data(2020, 2026) # Ingest standard subset window
    ingestor.ingest_players()
    ingestor.ingest_matches()