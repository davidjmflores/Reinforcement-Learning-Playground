from environments.ch2_nonstationary_karmedbandit import NonstationaryKArmedBandit
from algorithms.alpha_epsilon_greedy import AlphaEpsilonGreedyAgent
from algorithms.epsilon_greedy import EpsilonGreedyAgent

import numpy as np
import matplotlib.pyplot as plt


k = 10
epsilon = 0.1
alpha = 0.1
T = 10000
runs = 2000
base_seed = 0

# Exercise 2.5 env params
init_val = 0.0 # initial value for all estimated action values
sigma_r = 1.0 # reward std dev
sigma_n = 0.01 # random walk std dev

def run(agent_run, label):
    reward_sum = np.zeros(T, dtype=float)
    optimal_count = np.zeros(T, dtype=int)

    for run_i in range(runs):
        env_seed = base_seed + run_i
        agent_seed = base_seed + 10000 + run_i

        env = NonstationaryKArmedBandit(k=k, init_val=init_val,sigma_r=sigma_r, sigma_n=sigma_n, seed=env_seed)
        env.reset()

        agent = agent_run()
        agent_rng = np.random.default_rng(agent_seed)

        for t in range(T):
            a = agent.select_action(agent_rng)
            r, info = env.step(a)
            agent.update(a, r)
            
            reward_sum[t] += r
            optimal_count[t] += int(info["is_optimal"])

    avg_reward = reward_sum / runs
    opt_frac = optimal_count / runs
    return avg_reward, opt_frac, label

avg_sa, opt_sa, lab_sa = run(
    lambda: EpsilonGreedyAgent(k=k, epsilon=epsilon),
    label=f"ε={epsilon}, sample-average"
)

avg_a, opt_a, lab_a = run(
    lambda: AlphaEpsilonGreedyAgent(k=k, epsilon=epsilon, alpha=alpha),
    label=f"ε={epsilon}, α={alpha}"
)

steps = np.arange(T)

# Average reward
plt.figure()
plt.plot(steps, avg_sa, label=lab_sa)
plt.plot(steps, avg_a, label=lab_a)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Exercise 2.5: Nonstationary 10-armed bandit")
plt.grid(True)
plt.legend()
plt.show()

# % optimal action
plt.figure()
plt.plot(steps, 100.0 * opt_sa, label=lab_sa)
plt.plot(steps, 100.0 * opt_a, label=lab_a)
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Exercise 2.5: % Optimal Action")
plt.grid(True)
plt.legend()
plt.show()

print("Final avg reward (sample-average):", avg_sa[-1])
print("Final % optimal (sample-average):", 100.0 * opt_sa[-1])
print("Final avg reward (alpha):", avg_a[-1])
print("Final % optimal (alpha):", 100.0 * opt_a[-1])