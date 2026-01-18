from envs.stationary_karmedbandit import StationaryKArmedBandit
from algs.epsilon_greedy import EpsilonGreedyAgent
from algs.upper_confidence_bound import UpperConfidenceBound

import numpy as np
import matplotlib.pyplot as plt

k = 10
T = 1000
runs = 2000
base_seed = 0
c = 2.0
epsilon = 0.1

def run(agent_run, label):
    reward_sum = np.zeros(T, dtype=float)
    optimal_count = np.zeros(T, dtype=int)

    for run_i in range(runs):
        env_seed = base_seed + run_i
        agent_seed = base_seed + 10000 + run_i

        env = StationaryKArmedBandit(k=k, mean_q=0.0, sigma_q=1.0, sigma_r=1.0, seed=env_seed)
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

avg_ucb, opt_ucb, label_ucb = run(
    lambda: UpperConfidenceBound(k=k, c=c),
    label=f"c={c}"
)

avg_eg, opt_eg, label_eg = run(
    lambda: EpsilonGreedyAgent(k=k, epsilon=epsilon),
    label=f"ε={epsilon}"
)

steps = np.arange(T)

# Average reward
plt.figure()
plt.plot(steps, avg_ucb, label=label_ucb)
plt.plot(steps, avg_eg, label=label_eg)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("UCB vs. ε-Greedy: Average Reward")
plt.grid(True)
plt.legend()
plt.show()

# % optimal action
plt.figure()
plt.plot(steps, 100.0 * opt_ucb, label=label_ucb)
plt.plot(steps, 100.0 * opt_eg, label=label_eg)
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("UCB vs. ε-Greedy: % Optimal Action")
plt.grid(True)
plt.legend()
plt.show()