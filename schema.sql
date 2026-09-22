-- Disable foreign key checks during initialization
PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS rolling_features;
DROP TABLE IF EXISTS elo_history;
DROP TABLE IF EXISTS player_match_stats;
DROP TABLE IF EXISTS matches;
DROP TABLE IF EXISTS players;
DROP TABLE IF EXISTS model_versions;

PRAGMA foreign_keys = ON;

-- 1. PLAYERS TABLE
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    hand TEXT CHECK(hand IN ('R', 'L', 'U')), -- Right, Left, Unknown
    birth_date DATE,
    country_code TEXT,
    height_cm REAL
);

-- 2. MATCHES TABLE
CREATE TABLE matches (
    match_id TEXT PRIMARY KEY,
    tour_year INTEGER NOT NULL,
    tour_date DATE NOT NULL,
    tournament_name TEXT NOT NULL,
    tournament_level TEXT CHECK(tournament_level IN ('G', 'M', 'A', 'C', 'S', 'F', 'D')), -- Grand Slams, Masters, ATP250/500, Challengers, etc.
    surface TEXT NOT NULL CHECK(surface IN ('Hard', 'Clay', 'Grass', 'Carpet')),
    indoor_flag INTEGER DEFAULT 0,
    best_of INTEGER DEFAULT 3 CHECK(best_of IN (3, 5)),
    round TEXT NOT NULL,
    winner_id INTEGER NOT NULL,
    loser_id INTEGER NOT NULL,
    score_string TEXT NOT NULL,
    winner_rank INTEGER,
    winner_rank_points INTEGER,
    loser_rank INTEGER,
    loser_rank_points INTEGER,
    match_minutes INTEGER,
    FOREIGN KEY(winner_id) REFERENCES players(player_id),
    FOREIGN KEY(loser_id) REFERENCES players(player_id)
);

-- 3. PLAYER MATCH STATISTICS (Normalized detailed metrics per player per match)
CREATE TABLE player_match_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    opponent_id INTEGER NOT NULL,
    is_winner INTEGER NOT NULL CHECK(is_winner IN (0, 1)),
    
    -- Serve Statistics
    aces INTEGER,
    double_faults INTEGER,
    service_points_played INTEGER,
    first_serves_in INTEGER,
    first_serve_points_won INTEGER,
    second_serve_points_won INTEGER,
    service_games_played INTEGER,
    break_points_saved INTEGER,
    break_points_faced INTEGER,

    -- Derived Return Statistics (Opponent's Service Performance)
    return_points_played INTEGER,
    return_points_won INTEGER,
    first_return_points_won INTEGER,
    second_return_points_won INTEGER,
    break_points_converted INTEGER,
    break_points_opportunities INTEGER,
    return_games_played INTEGER,

    FOREIGN KEY(match_id) REFERENCES matches(match_id),
    FOREIGN KEY(player_id) REFERENCES players(player_id),
    FOREIGN KEY(opponent_id) REFERENCES players(player_id)
);

-- 4. ELO HISTORY (Pre-match snapshots to eliminate data leakage)
CREATE TABLE elo_history (
    elo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    pre_match_overall_elo REAL NOT NULL,
    post_match_overall_elo REAL NOT NULL,
    pre_match_surface_elo REAL NOT NULL,
    post_match_surface_elo REAL NOT NULL,
    surface TEXT NOT NULL,
    FOREIGN KEY(match_id) REFERENCES matches(match_id),
    FOREIGN KEY(player_id) REFERENCES players(player_id)
);

-- 5. ROLLING FEATURES (Pre-match chronological snapshot)
CREATE TABLE rolling_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    
    -- Rolling Form Metrics
    win_pct_last_5 REAL,
    win_pct_last_10 REAL,
    win_pct_last_20 REAL,
    surface_win_pct_last_10 REAL,
    
    -- Opponent-Adjusted Serve & Return Strengths
    adj_serve_strength REAL,
    adj_return_strength REAL,
    
    -- Fatigue & Workload Metrics
    rest_days REAL,
    matches_last_7_days INTEGER,
    matches_last_14_days INTEGER,
    sets_last_14_days INTEGER,
    minutes_last_14_days INTEGER,

    FOREIGN KEY(match_id) REFERENCES matches(match_id),
    FOREIGN KEY(player_id) REFERENCES players(player_id)
);

-- 6. PREDICTIONS (Out-of-sample and inference store)
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    player_a_id INTEGER NOT NULL,
    player_b_id INTEGER NOT NULL,
    p_player_a_win REAL NOT NULL,
    p_player_b_win REAL NOT NULL,
    p_2_0 REAL NOT NULL,
    p_2_1 REAL NOT NULL,
    p_1_2 REAL NOT NULL,
    p_0_2 REAL NOT NULL,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(match_id) REFERENCES matches(match_id)
);

-- 7. MODEL VERSIONS
CREATE TABLE model_versions (
    version_id TEXT PRIMARY KEY,
    trained_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_cutoff_date DATE NOT NULL,
    brier_score REAL,
    log_loss REAL,
    exact_score_accuracy REAL,
    parameters_json TEXT
);

-- INDEXES FOR FAST CHRONOLOGICAL FETCHING AND JOINING
CREATE INDEX idx_matches_date ON matches(tour_date);
CREATE INDEX idx_matches_winner ON matches(winner_id);
CREATE INDEX idx_matches_loser ON matches(loser_id);
CREATE INDEX idx_stats_player_match ON player_match_stats(player_id, match_id);
CREATE INDEX idx_elo_player_match ON elo_history(player_id, match_id);
CREATE INDEX idx_rolling_player_match ON rolling_features(player_id, match_id);