



import numpy as np
from src.environment import AbruptSlightlyNonstationaryEnvironment


env = AbruptSlightlyNonstationaryEnvironment(
    mean=[[10, 20], [20, 10], [15, 15]],  # Valuation means for each item
    std=[[2, 2], [2, 2], [2, 2]],      # Standard deviations for each item
    time_horizon=30
)

print(env.valuations.shape)

n_items = env.valuations.shape[0]
n_rounds = env.valuations.shape[1]

# print(env.valuations)

l = []
for item in range(n_items):
    for round in range(n_rounds):
        l.append((float(env.valuations[item, round]), item, round))

l.sort(key=lambda x: x[0], reverse=True)

# matrice di uni con righe il numero di item e colonne i round
prices = np.ones((n_items, n_rounds))
prices *= float('inf')

budget = 10

for i in range(budget):
    v, item, round = l[i]
    prices[item, round] = v

print(prices)

arms = list(range(0, 40, 1))

arm_indexes_to_play = np.ones((n_items, n_rounds)) * (-1)

for item in range(n_items):
    for round in range(n_rounds):
        if prices[item, round] == float('inf'):
            continue
        # binsearch prices[item, round] in arms using the library
        idx = np.searchsorted(arms, prices[item, round] + 1e-9) - 1
        assert idx >= 0, "Il valore generato dall'env è più basso della nostra arm più bassa"
        arm_indexes_to_play[item, round] = idx

print(arm_indexes_to_play)   
        
