import numpy as np

"""
This function create a transition matrix based on:
    - min: Minimum value of chips that the player can reach. When the player reaches this amount, he withdraws. 
    - max: Maximun value of chips that the player can reach. When the player reaches this amount, he withdraws.
    - bets: Array of tuples representing all possible bets. Each tuple is an apuetas. The first component of the
            tuple is the profit obtained per chip bet. The second component is the probability of winning the bet. 
    - p_bets: Array of odds that the player will place a certain bet. This array has as many components as bets have.          
"""


def create_transition_matrix(min, max, bets, p_bets):
    if len(bets) != len(p_bets):
        print("Longitude of the possible bets and probabilities of bets are not the same.")
        return None
    size = max - min + bets[-1][0]
    matrix = np.zeros((size, size,), dtype=float)
    matrix[min][min] = 1
    for i in range(size - 1 - bets[-1][0]):
        i = i + 1
        p_lose = 0
        for bet, p_bet in zip(bets, p_bets):
            matrix[i][i + bet[0]] = bet[1] * p_bet
            p_lose += (1 - bet[1]) * p_bet
        matrix[i][i - 1] = p_lose
    for i in range(bets[-1][0]):
        matrix[i + max][i + max] = 1
    return matrix
