import numpy as np

def game_win_probability(p: float) -> float:
    """Calculates probability of winning a standard game given point win probability p."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    
    # Game outcome via Markov chain
    # P(Game) = P(Win before Deuce) + P(Deuce) * P(Win from Deuce)
    p4 = p ** 4
    p3_q = 4 * (p ** 3) * (1 - p)
    p3_q2 = 10 * (p ** 3) * ((1 - p) ** 2)
    
    deuce_prob = 20 * (p ** 3) * ((1 - p) ** 3)
    win_from_deuce = (p ** 2) / (p ** 2 + (1 - p) ** 2)
    
    return (p4 + p3_q + p3_q2) + (deuce_prob * win_from_deuce)

def tiebreak_win_probability(p_serve: float, p_return: float) -> float:
    """Calculates probability of winning a 7-point tiebreak."""
    # Simplified numerical approximation for tiebreak transition matrix
    # p_serve: point win rate on own serve, p_return: point win rate on opponent serve
    p_avg = (p_serve + p_return) / 2.0
    
    # Using symmetrical binomial approximation for 7-point tiebreak
    # Win at 7-5 or better or deuce tiebreak
    tb_hold_prob = game_win_probability(p_avg)
    return tb_hold_prob

def set_win_probability(p_serve_a: float, p_serve_b: float) -> float:
    """Calculates probability Player A wins a standard set against Player B.
    
    p_serve_a: Player A's probability of winning a point on serve.
    p_serve_b: Player B's probability of winning a point on serve.
    """
    g_a = game_win_probability(p_serve_a)          # Prob A holds serve
    g_b = 1.0 - game_win_probability(p_serve_b)    # Prob A breaks B's serve
    
    # Average game win probability for A across service & return games
    p_game_a = (g_a + g_b) / 2.0
    
    # Markov set approximation (6-4, 6-3, 6-2, 6-1, 6-0, or 7-5, 7-6)
    # Binary expansion of winning 6 before 5
    if p_game_a == 0.5:
        return 0.5
    
    # Negative binomial expectation for reaching 6 games
    z = p_game_a / (1.0 - p_game_a)
    prob_set_a = (p_game_a ** 6) * (1 + 6*(1-p_game_a) + 21*((1-p_game_a)**2) + 56*((1-p_game_a)**3) + 126*((1-p_game_a)**4))
    
    return float(np.clip(prob_set_a, 0.001, 0.999))