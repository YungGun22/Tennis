import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from typing import Tuple, Dict, Any
import joblib
import os

class SetWinModelEngine:
    """Trains and evaluates LightGBM binary classifier for set-level outcomes."""

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.feature_columns = [
            "elo_diff",
            "surface_elo_diff",
            "win_pct_last_10_diff",
            "surface_win_pct_last_10_diff",
            "adj_serve_strength_diff",
            "adj_return_strength_diff",
            "rest_days_diff",
            "matches_last_14_days_diff"
        ]

    def prepare_differential_dataset(self, df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Transforms raw match features into Player A - Player B differential pairs."""
        X = pd.DataFrame()
        
        X["elo_diff"] = df_features["p1_elo"] - df_features["p2_elo"]
        X["surface_elo_diff"] = df_features["p1_surface_elo"] - df_features["p2_surface_elo"]
        X["win_pct_last_10_diff"] = df_features["p1_win_pct_last_10"] - df_features["p2_win_pct_last_10"]
        X["surface_win_pct_last_10_diff"] = df_features["p1_surface_win_pct_last_10"] - df_features["p2_surface_win_pct_last_10"]
        X["adj_serve_strength_diff"] = df_features["p1_adj_serve_strength"] - df_features["p2_adj_serve_strength"]
        X["adj_return_strength_diff"] = df_features["p1_adj_return_strength"] - df_features["p2_adj_return_strength"]
        X["rest_days_diff"] = df_features["p1_rest_days"] - df_features["p2_rest_days"]
        X["matches_last_14_days_diff"] = df_features["p1_matches_last_14_days"] - df_features["p2_matches_last_14_days"]
        
        y = df_features["p1_won_set"]
        return X[self.feature_columns], y

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, float]:
        """Trains LightGBM model with Sigmoid Platt Calibration."""
        print("[SET MODEL] Training LightGBM Base Set-Win Predictor...")
        
        base_lgb = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        base_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        # Calibrate probabilities
        print("[SET MODEL] Calibrating probabilities via Platt Scaling...")
        self.model = CalibratedClassifierCV(estimator=base_lgb, method="sigmoid", cv="prefit")
        self.model.fit(X_val, y_val)

        # Validation Metrics
        val_preds = self.model.predict_proba(X_val)[:, 1]
        loss = log_loss(y_val, val_preds)
        brier = brier_score_loss(y_val, val_preds)
        
        print(f"[METRICS] Set Model Log-Loss: {loss:.4f} | Brier Score: {brier:.4f}")
        
        # Save Artifact
        model_path = os.path.join(self.model_dir, "set_win_model.joblib")
        joblib.dump(self.model, model_path)
        print(f"[SAVED] Set win model saved to {model_path}")

        return {"log_loss": float(loss), "brier_score": float(brier)}

    def predict_set_probability(self, feature_diff_dict: Dict[str, float]) -> float:
        """Predicts probability of Player 1 winning a set."""
        if self.model is None:
            model_path = os.path.join(self.model_dir, "set_win_model.joblib")
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                raise FileNotFoundError("Set model artifact not found. Train model first.")

        df_input = pd.DataFrame([feature_diff_dict])[self.feature_columns]
        prob = self.model.predict_proba(df_input)[0, 1]
        return float(prob)