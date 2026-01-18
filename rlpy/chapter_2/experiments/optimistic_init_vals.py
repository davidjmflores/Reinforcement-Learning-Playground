from envs.nonstationary_karmedbandit import NonstationaryKArmedBandit
from algs.epsilon_greedy import EpsilonGreedyAgent
from algs.greedy import GreedyAgent

import numpy as np
import matplotlib.pyplot as plt

k = 10
T = 1000
runs = 2000
base_seed = 0
sigma_r = 1.0 # reward std dev
alpha = 0.1

def run(agent_run, label):
    reward_sum = np.zeros(T, dtype=float)
    optimal_count = np.zeros(T, dtype=int)

    for run_i in range(runs):
        env_seed = base_seed + run_i
        agent_seed = base_seed + 10000 + run_i

        env = NonstationaryKArmedBandit(k=k, mean_q=0.0, sigma_q=1.0, sigma_r=sigma_r, alpha=alpha, seed=env_seed)
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

avg_opt, opt_opt, lab_opt = run(
    lambda: GreedyAgent(k=k, q_init=5.0),
    label=f"ε={0.0}, Q_1={5.0}"
)

avg_rlstc, opt_rlstc, lab_rlstc = run(
    lambda: EpsilonGreedyAgent(k=k, epsilon=0.1, q_init=0.0),
    label=f"ε={0.1}, Q_1={0.0}"
)

steps = np.arange(T)

# Average reward
plt.figure()
plt.plot(steps, avg_opt, label=lab_opt)
plt.plot(steps, avg_rlstc, label=lab_rlstc)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Optimistic vs. Realistic: Average Reward")
plt.grid(True)
plt.legend()
plt.show()

# % optimal action
plt.figure()
plt.plot(steps, 100.0 * opt_opt, label=lab_opt)
plt.plot(steps, 100.0 * opt_rlstc, label=lab_rlstc)
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Optimistic vs. Realistic: % Optimal Action")
plt.grid(True)
plt.legend()
plt.show()