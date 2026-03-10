import numpy as np
import matplotlib.pyplot as plt

from environments.ch_6_cliff_walking import CliffWalking
from algorithms.td.sarsa import Sarsa
from algorithms.td.q_learning import QLearning

episodes = 500
gamma = 1.0
alpha = 0.5
epsilon = 0.1

num_seeds = 1000
base_seed = 0

sarsa_returns_all = np.zeros((num_seeds, episodes), dtype=float)
q_returns_all     = np.zeros((num_seeds, episodes), dtype=float)

for i in range(num_seeds):
    rng = np.random.default_rng(base_seed + i)

    env = CliffWalking()

    sarsa = Sarsa(env, gamma, alpha, epsilon, rng)
    q_learning = QLearning(env, rng, alpha, epsilon, gamma)

    _, sarsa_returns = sarsa.run(episodes)
    _, q_returns = q_learning.run(episodes)

    sarsa_returns_all[i, :] = np.asarray(sarsa_returns, dtype=float)
    q_returns_all[i, :]     = np.asarray(q_returns, dtype=float)

sarsa_mean = sarsa_returns_all.mean(axis=0)
q_mean     = q_returns_all.mean(axis=0)

sarsa_se = sarsa_returns_all.std(axis=0, ddof=1) / np.sqrt(num_seeds)
q_se     = q_returns_all.std(axis=0, ddof=1) / np.sqrt(num_seeds)

x = np.arange(1, episodes + 1)

plt.figure()
plt.plot(x, sarsa_mean, label="Sarsa (mean over seeds)")
plt.plot(x, q_mean, label="Q-learning (mean over seeds)")

plt.fill_between(x, sarsa_mean - sarsa_se, sarsa_mean + sarsa_se, alpha=0.2)
plt.fill_between(x, q_mean - q_se, q_mean + q_se, alpha=0.2)
plt.ylim(-100, 0)

plt.xlabel("Episodes")
plt.ylabel("Sum of rewards during episode")
plt.title("Cliff Walking (avg over 1000 seeds)")
plt.grid(True)
plt.legend()
plt.show()