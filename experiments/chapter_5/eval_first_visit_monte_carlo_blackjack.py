from environments.ch5_blackjack import Blackjack
from algorithms.first_visit_mc_prediction import FirstVisitMCPrediction
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class Stick20Policy:
    def pi(self, s):
        player_sum = s[0]
        if player_sum in (20, 21): return {0: 1.0, 1: 0.0}
        else: return {0: 0.0, 1: 1.0}

episodes = 500000
gamma = 1
gen = 0
rng = np.random.default_rng(gen)

env = Blackjack()
policy = Stick20Policy()
alg = FirstVisitMCPrediction(env, policy, v0 = 0, gamma = gamma)

V = alg.prediction(episodes, rng)

dealer_vals = np.arange(1, 11) 
player_vals = np.arange(12, 22)    
X, Y = np.meshgrid(dealer_vals, player_vals)

def build_surface(V, usable_ace):
    Z = np.zeros_like(X, dtype=float)
    for i, ps in enumerate(player_vals):
        for j, dc in enumerate(dealer_vals):
            Z[i, j] = V[(ps, dc, usable_ace)]
    return Z

Z_usable = build_surface(V, usable_ace=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z_usable, cmap="viridis")

ax.set_xlabel("Dealer showing")
ax.set_ylabel("Player sum")
ax.set_zlabel("Value")
ax.set_title("Value function (usable ace)")

plt.show()

Z_no_usable = build_surface(V, usable_ace=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z_no_usable, cmap="viridis")

ax.set_xlabel("Dealer showing")
ax.set_ylabel("Player sum")
ax.set_zlabel("Value")
ax.set_title("Value function (no usable ace)")

plt.show()



