import matplotlib.pyplot as plt
import numpy as np

from environments.ch8_blocking_maze import BlockingMaze
from algorithms.planning_and_learning.dyna_q import DynaQ
from algorithms.planning_and_learning.dyna_q_plus import DynaQPlus

episodes = 100
epsilon = 0.1
gamma = 0.95
alpha = 0.1
base_seed = 0
n = 50
kappa = 0.1

env = BlockingMaze()
plt.figure()

rng_dynaq = np.random.default_rng(base_seed)
rng_dynaqplus = np.random.default_rng(base_seed + 1)
DynaQAgent = DynaQ(rng=rng_dynaq, env=env, epsilon=epsilon, gamma=gamma, alpha=alpha, n=n)
DynaQPlusAgent = DynaQPlus(rng=rng_dynaqplus, env=env, epsilon=epsilon, gamma=gamma, alpha=alpha, n=n, kappa=kappa)

max_steps = 3000

_, dynaq_cumulative_reward = DynaQAgent.run(episodes)
_, dynaqplus_cumulative_reward = DynaQPlusAgent.run(episodes)

dynaq_cumulative_reward = dynaq_cumulative_reward[:max_steps]
dynaqplus_cumulative_reward = dynaqplus_cumulative_reward[:max_steps]

x = np.arange(1, max_steps + 1)

plt.plot(x, dynaq_cumulative_reward, label="Dyna-Q")
plt.plot(x, dynaqplus_cumulative_reward, label="Dyna-Q+")
plt.xlabel("Time steps")
plt.ylabel("Cumulative Reward")
plt.title("Exercise 8.2: Blocking Maze")
plt.grid(True)
plt.legend()
plt.show()
