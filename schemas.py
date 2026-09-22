from pydantic import BaseModel, Field
from typing import Dict, Optional, List

class MatchPredictionRequest(BaseModel):
    player_a_id: int = Field(..., example=100644, description="ATP/WTA ID for Player A")
    player_b_id: int = Field(..., example=104925, description="ATP/WTA ID for Player B")
    surface: str = Field(..., example="Hard", description="Surface type: Hard, Clay, Grass, Carpet")
    best_of: int = Field(default=3, example=3, description="Match format: 3 or 5 sets")
    simulations: int = Field(default=10000, ge=1000, le=100000, description="Monte Carlo iterations")
    match_date: Optional[str] = Field(default=None, example="2026-09-22", description="Date string YYYY-MM-DD")

class ExactScoreDistribution(BaseModel):
    score_probabilities: Dict[str, float] = Field(..., example={"2-0": 0.421, "2-1": 0.283, "0-2": 0.182, "1-2": 0.114})
    most_likely_score: str = Field(..., example="2-0")
    confidence: float = Field(..., example=0.421)

class MatchPredictionResponse(BaseModel):
    status: str = Field(default="success")
    player_a_id: int
    player_b_id: int
    player_a_win_probability: float
    player_b_win_probability: float
    single_set_win_prob_a: float
    exact_scores: ExactScoreDistribution
    expected_total_sets: float