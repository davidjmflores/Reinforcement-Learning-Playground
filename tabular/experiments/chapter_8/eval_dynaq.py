import numpy as np
import matplotlib.pyplot as plt

from environments.ch_8_dyna_maze import DynaMaze
from algorithms.planning_and_learning.dyna_q import DynaQ

episodes = 50
epsilon = 0.1
gamma = 0.95
alpha = 0.1
planning_steps = [0, 5, 50] # number of planning steps
base_seed = 0

env = DynaMaze()
plt.figure()
x = np.arange(1, episodes + 1)

for n in planning_steps:
    rng = np.random.default_rng(base_seed + n)
    agent = DynaQ(rng=rng, env=env, epsilon=epsilon, gamma=gamma, alpha=alpha, n=n)

    _, steps_per_episode = agent.run(episodes)
    plt.plot(x, steps_per_episode, label=f'{n} planning steps')

plt.xlabel("Episodes")
plt.ylabel("Steps per episode")
plt.title("Exercise 8.1: Dyna Maze")
plt.grid(True)
plt.legend()
plt.show()


