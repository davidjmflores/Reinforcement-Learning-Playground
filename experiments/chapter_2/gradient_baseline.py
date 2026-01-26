from environments.ch2_stationary_karmedbandit import StationaryKArmedBandit
from algorithms.gradient import GradientBanditAgent

import numpy as np
import matplotlib.pyplot as plt

k = 10
T = 1000
runs = 2000
base_seed = 0
alpha = 0.1

def run(agent_run, label):
    reward_sum = np.zeros(T, dtype=float)
    optimal_count = np.zeros(T, dtype=int)

    for run_i in range(runs):
        env_seed = base_seed + run_i
        agent_seed = base_seed + 10000 + run_i

        env = StationaryKArmedBandit(k=k, mean_q=4.0, sigma_q=1.0, sigma_r=1.0, seed=env_seed)
        env.reset()

        agent = agent_run()
        agent_rng = np.random.default_rng(agent_seed)

        for t in range(T):
            a = agent.select_action(agent_rng)
            r, info = env.step(a)
            agent.update(a,r)

            reward_sum[t] += r
            optimal_count[t] += int(info["is_optimal"])
    
    avg_reward = reward_sum / runs
    opt_frac = optimal_count / runs
    return avg_reward, opt_frac, label

avg_bl, opt_bl, label_bl = run(
    lambda: GradientBanditAgent(k=k, alpha=alpha, use_bl = True),
    label=f"With baseline, alpha = {alpha}"
)

avg_wbl, opt_wbl, label_wbl = run(
    lambda: GradientBanditAgent(k=k, alpha=alpha, use_bl = False),
    label=f"Without baseline, alpha = {alpha}"
)

steps = np.arange(T)

# Average reward
plt.figure()
plt.plot(steps, avg_bl, label=label_bl)
plt.plot(steps, avg_wbl, label=label_wbl)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Gradient Bandit Algorithm: Average Reward")
plt.grid(True)
plt.legend()
plt.show()

# % optimal action
plt.figure()
plt.plot(steps, 100.0 * opt_bl, label=label_bl)
plt.plot(steps, 100.0 * opt_wbl, label=label_wbl)
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Gradient Bandit Algorithm: % Optimal Action")
plt.grid(True)
plt.legend()
plt.show()