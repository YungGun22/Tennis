import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DATA_DIR / "tennis.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

# Data Source Endpoint Templates (Jeff Sackmann Repositories)
DATA_SOURCES = {
    "atp_matches": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
    "atp_players": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv",
    "atp_rankings_current": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv",
    "wta_matches": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
    "wta_players": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_players.csv"
}

# Historical ingestion window
START_YEAR = 2000
END_YEAR = 2026

# Model & Feature Parameters
ELO_BASE = 1500.0
ELO_DEFAULT_K = 32.0
SURFACE_TYPES = ["Hard", "Clay", "Grass", "Carpet"]

# Temporal splits
TEMPORAL_VALIDATION_SPLIT_YEAR = 2023
TEMPORAL_TEST_SPLIT_YEAR = 2024

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)