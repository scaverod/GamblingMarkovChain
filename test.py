import transition_matrix as tm
import numpy as np
import visualize as vl

LEAVE_WITH_MIN = 0
LEAVE_WITH_MAX = 150

bets = [(1, float(18 / 37)),
        (2, float(12 / 37)),
        (5, float(6 / 37)),
        (8, float(4 / 37)),
        (11, float(3 / 37)),
        (17, float(2 / 37)),
        (35, float(1 / 37))]

p_bets = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.025, 0.025])
p_bets_eq = [float(1/len(bets))]*len(bets)
print(sum(p_bets_eq))

tmat = tm.create_transition_matrix(LEAVE_WITH_MIN, LEAVE_WITH_MAX, bets, p_bets)

print("Resultant transition matrix")
vl.print_matrix(tmat)

print("Test")
print(tmat.sum(1))
