import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from typing import Dict, List, Any

def compute_multiclass_brier_score(y_true_onehot: np.ndarray, y_prob_matrix: np.ndarray) -> float:
    """Computes the Multi-Class Brier Score across all exact score outcomes.
    
    Formula: BS = (1/N) * sum_{i=1}^N sum_{k=1}^K (f_ik - o_ik)^2
    """
    return float(np.mean(np.sum((y_prob_matrix - y_true_onehot) ** 2, axis=1)))

def evaluate_exact_score_predictions(
    y_true_scores: List[str],
    pred_prob_dicts: List[Dict[str, float]],
    possible_scores: List[str] = ["2-0", "2-1", "0-2", "1-2"]
) -> Dict[str, float]:
    """Computes exact match score accuracy, top-2 coverage, multi-class log-loss, and Brier score."""
    n_samples = len(y_true_scores)
    if n_samples == 0:
        return {}

    correct_top1 = 0
    correct_top2 = 0
    
    y_true_onehot = np.zeros((n_samples, len(possible_scores)))
    y_prob_matrix = np.zeros((n_samples, len(possible_scores)))

    score_to_idx = {s: i for i, s in enumerate(possible_scores)}

    for i, (true_score, prob_dict) in enumerate(zip(y_true_scores, pred_prob_dicts)):
        if true_score in score_to_idx:
            y_true_onehot[i, score_to_idx[true_score]] = 1.0

        # Extract probability vector
        probs = [prob_dict.get(s, 0.0) for s in possible_scores]
        total_p = sum(probs)
        if total_p > 0:
            probs = [p / total_p for p in probs] # Normalize
        y_prob_matrix[i, :] = probs

        # Rank predictions
        ranked_scores = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top1_score = ranked_scores[0][0] if len(ranked_scores) > 0 else None
        top2_scores = [x[0] for x in ranked_scores[:2]]

        if true_score == top1_score:
            correct_top1 += 1
        if true_score in top2_scores:
            correct_top2 += 1

    top1_acc = correct_top1 / n_samples
    top2_acc = correct_top2 / n_samples
    mc_log_loss = log_loss(y_true_onehot, y_prob_matrix)
    mc_brier = compute_multiclass_brier_score(y_true_onehot, y_prob_matrix)

    return {
        "sample_size": n_samples,
        "exact_score_top1_accuracy": round(top1_acc, 4),
        "exact_score_top2_accuracy": round(top2_acc, 4),
        "multiclass_log_loss": round(float(mc_log_loss), 4),
        "multiclass_brier_score": round(mc_brier, 4)
    }