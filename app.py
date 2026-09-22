from fastapi import FastAPI, HTTPException, status
import uvicorn
from src.api.schemas import MatchPredictionRequest, MatchPredictionResponse
from src.api.inference import RealTimeInferenceEngine

app = FastAPI(
    title="Tennis Exact Match Score Prediction Engine",
    description="Production REST API executing ML-backed Monte Carlo simulations for exact score tennis forecasting.",
    version="1.0.0"
)

# Global inference engine instance
inference_engine = RealTimeInferenceEngine()

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "service": "tennis-prediction-api"}

@app.post("/predict", response_model=MatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_match_exact_score(request: MatchPredictionRequest):
    """Predicts exact match score probability distribution for two players."""
    try:
        prediction = inference_engine.predict_match(
            player_a_id=request.player_a_id,
            player_b_id=request.player_b_id,
            surface=request.surface,
            best_of=request.best_of,
            simulations=request.simulations
        )
        return prediction
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)