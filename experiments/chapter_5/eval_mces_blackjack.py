from environments.ch5_blackjack import Blackjack
from policies.tabular_stochastic_policy import TabularStochasticPolicy
from algorithms.mc_es import MCES

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

episodes = 500000
gamma = 1.0
gen = 0
rng = np.random.default_rng(gen)

env = Blackjack()
policy = TabularStochasticPolicy(env)
alg = MCES(env, policy, gamma=1.0)

Q, pi = alg.exploring_starts(episodes, rng)

dealer_vals = np.arange(1, 11) 
player_vals = np.arange(12, 22)    
X, Y = np.meshgrid(dealer_vals, player_vals)

Pi_usable = np.zeros_like(X, dtype=int)
Pi_nousable = np.zeros_like(X, dtype=int)
V_usable = np.zeros_like(X, dtype=float)
V_nousable = np.zeros_like(X, dtype=float)

for i, ps in enumerate(player_vals):
    for j, ds in enumerate(dealer_vals):
        for ua, Pi_grid, V_grid in [
            (True, Pi_usable, V_usable),
            (False, Pi_nousable, V_nousable),
        ]:
            s = (ps, ds, ua)

            # value from Q
            V_grid[i, j] = max(Q[s].values())

            # policy action: either from Q greedy or from pi
            # (use pi if you want to trust your stored policy)
            dist = pi.pi(s)
            Pi_grid[i, j] = max(dist, key=dist.get)
            # or: Pi_grid[i, j] = max(Q[s], key=Q[s].get)

fig, ax = plt.subplots(2, 1, figsize=(6, 8))

ax[0].imshow(Pi_usable, origin="lower", aspect="auto")
ax[0].set_title("Usable ace")
ax[0].set_xticks(np.arange(10)); ax[0].set_xticklabels(dealer_vals)
ax[0].set_yticks(np.arange(10)); ax[0].set_yticklabels(player_vals)
ax[0].set_xlabel("Dealer showing"); ax[0].set_ylabel("Player sum")

ax[1].imshow(Pi_nousable, origin="lower", aspect="auto")
ax[1].set_title("No usable ace")
ax[1].set_xticks(np.arange(10)); ax[1].set_xticklabels(dealer_vals)
ax[1].set_yticks(np.arange(10)); ax[1].set_yticklabels(player_vals)
ax[1].set_xlabel("Dealer showing"); ax[1].set_ylabel("Player sum")

fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(2, 1, 1, projection="3d")
ax1.plot_surface(X, Y, V_usable)
ax1.set_title("V* (Usable ace)")
ax1.set_xlabel("Dealer showing")
ax1.set_ylabel("Player sum")
ax1.set_zlabel("Value")

ax2 = fig.add_subplot(2, 1, 2, projection="3d")
ax2.plot_surface(X, Y, V_nousable)
ax2.set_title("V* (No usable ace)")
ax2.set_xlabel("Dealershowing")
ax2.set_ylabel("Player sum")
ax2.set_zlabel("Value")

plt.tight_layout()
plt.show()
