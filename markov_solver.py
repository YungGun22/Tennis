from typing import Dict

def solve_exact_match_probabilities(p_set_a: float, best_of_five: bool = False) -> Dict[str, float]:
    """Calculates exact analytical score probabilities for best-of-3 or best-of-5 matches.
    
    p_set_a: Probability Player A wins any given set (assumed i.i.d. baseline).
    """
    p = float(p_set_a)
    q = 1.0 - p

    if not best_of_five:
        # Best of 3 format (First to 2 sets)
        p_2_0 = p ** 2
        p_2_1 = 2 * (p ** 2) * q
        p_0_2 = q ** 2
        p_1_2 = 2 * (q ** 2) * p

        return {
            "2-0": float(p_2_0),
            "2-1": float(p_2_1),
            "0-2": float(p_0_2),
            "1-2": float(p_1_2),
            "win_prob_a": float(p_2_0 + p_2_1),
            "win_prob_b": float(p_0_2 + p_1_2)
        }
    else:
        # Best of 5 format (First to 3 sets)
        p_3_0 = p ** 3
        p_3_1 = 3 * (p ** 3) * q
        p_3_2 = 6 * (p ** 3) * (q ** 2)
        p_0_3 = q ** 3
        p_1_3 = 3 * (q ** 3) * p
        p_2_3 = 6 * (q ** 3) * (p ** 2)

        return {
            "3-0": float(p_3_0),
            "3-1": float(p_3_1),
            "3-2": float(p_3_2),
            "0-3": float(p_0_3),
            "1-3": float(p_1_3),
            "2-3": float(p_2_3),
            "win_prob_a": float(p_3_0 + p_3_1 + p_3_2),
            "win_prob_b": float(p_0_3 + p_1_3 + p_2_3)
        }