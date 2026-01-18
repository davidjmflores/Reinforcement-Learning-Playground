from envs.stationary_karmedbandit import StationaryKArmedBandit
from algs.greedy import GreedyAgent
import numpy as np
import matplotlib.pyplot as plt

k = 10
mean_q = 0.0
sigma_q = 1.0
sigma_r = 1.0
T = 1000
runs = 2000
base_seed = 0

reward_sum = np.zeros(T, dtype=float)
optimal_count = np.zeros(T, dtype=int)

for run in range(runs):
    env_seed = base_seed + run
    agent_seed = base_seed + run

    env = StationaryKArmedBandit(k=k, mean_q=mean_q,sigma_q=sigma_q,sigma_r=sigma_r, seed=env_seed)
    agent = GreedyAgent(k=k)
    agent_rng = np.random.default_rng(agent_seed)

    env.reset()
    for t in range(T):
        a = agent.select_action(agent_rng)
        r, info = env.step(a)
        agent.update(a, r)
        
        reward_sum[t] += r
        optimal_count[t] += int(info["is_optimal"])

avg_reward = reward_sum / runs
opt_frac = optimal_count / runs

print("Final avg reward: ", avg_reward[-1])
print("Final % optimal: ", 100 * opt_frac[-1])

steps = np.arange(T)

# Avg Reward vs. Steps
plt.figure()
plt.plot(steps, avg_reward)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Greedy: Average Reward vs. Steps")
plt.grid(True)
plt.show()

# % Optimal Action vs. Steps
plt.figure()
plt.plot(steps, 100 * opt_frac)
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Greedy: % Optimal Action vs Steps")
plt.grid(True)
plt.show()
