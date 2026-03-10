import numpy as np
import matplotlib.pyplot as plt

from environments.ch6_windy_gridworld import WindyGridworld
from algorithms.td.sarsa import Sarsa
# from algorithms.mc.on_policy_fvmc import OnPolicyFVMC

episodes = 200
gamma = 1.0
alpha = 0.5
epsilon = 0.1
gen = 0
rng = np.random.default_rng(gen)
tolerance = 1e-3

env = WindyGridworld()
agent_td = Sarsa(env, gamma, alpha, epsilon, rng)
# agent_mc = OnPolicyFVMC(env, epsilon, gamma, tolerance, rng)

_, td_episode_end_steps = agent_td.run(episodes)
# _, _, mc_episode_end_steps = agent_mc.run(episodes)



td_x = td_episode_end_steps
td_y = np.arange(1, len(td_x) + 1)

# mc_x = mc_episode_end_steps
# mc_y = np.arange(1, len(mc_x) + 1)


plt.figure()
plt.plot(td_x, td_y, label="Sarsa")
# plt.plot(mc_x, mc_y, label="MC")
plt.xlabel("Time Steps")
plt.ylabel("Episodes")
plt.title("Windy Gridworld")
plt.grid(True)
plt.legend()
plt.show()


