from environments.ch2_nonstationary_karmedbandit import NonstationaryKArmedBandit

from algorithms.epsilon_greedy import EpsilonGreedyAgent
from algorithms.alpha_epsilon_greedy import AlphaEpsilonGreedyAgent
from algorithms.alpha_upper_confidence_bound import AlphaUpperConfidenceBound
from algorithms.gradient import GradientBanditAgent
from algorithms.alpha_greedy import AlphaGreedyAgent

import numpy as np
import matplotlib.pyplot as plt

k = 10
T = 200000
eval_start = 100000
runs = 20
alpha = 0.1
base_seed = 0

def run_one(agent_run, param_value):
    total_score = 0.0

    for run_i in range(runs):
        env_seed = base_seed + run_i
        agent_seed = base_seed + 10000 + run_i

        env = NonstationaryKArmedBandit(k=k, sigma_r=1.0, sigma_n=0.01, init_val=0.0, seed=env_seed)
        env.reset()

        agent = agent_run(param_value)
        agent_rng = np.random.default_rng(agent_seed)

        reward_sum = 0.0
        reward_count = 0

        for t in range(1, T+1):
            a = agent.select_action(agent_rng)
            r, info = env.step(a)
            agent.update(a,r)
            
            if t > eval_start: 
                reward_sum += r
                reward_count += 1
    
        total_score += (reward_sum / reward_count)

    return total_score / runs

def sweep(agent_run, grid):
    return np.array([run_one(agent_run, x) for x in grid], dtype=float)

steps_all = np.array([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4], dtype=float)
steps_eps = steps_all[steps_all <= 1.0]

y_eg_avg   = sweep(lambda eps: EpsilonGreedyAgent(k=k, epsilon=eps), steps_eps)
y_eg_const = sweep(lambda eps: AlphaEpsilonGreedyAgent(k=k, epsilon=eps, alpha=alpha), steps_eps)

y_grad = sweep(lambda a: GradientBanditAgent(k=k, alpha=a, use_bl=True), steps_all)
y_ucb  = sweep(lambda c: AlphaUpperConfidenceBound(k=k, c=c, alpha=alpha), steps_all)
y_opt  = sweep(lambda q0: AlphaGreedyAgent(k=k, q_init=q0, alpha=alpha), steps_all)

plt.figure()
plt.plot(steps_eps, y_eg_avg, label="ε-greedy (sample-average)")
plt.plot(steps_eps, y_eg_const, label=f"ε-greedy (constant α={alpha})")
plt.plot(steps_all, y_grad, label="gradient bandit")
plt.plot(steps_all, y_ucb, label=f"UCB (α={alpha})")
plt.plot(steps_all, y_opt, label=f"optimistic greedy (α={alpha})")

plt.xscale("log", base=2)
plt.xticks(steps_all, [str(x) for x in steps_all])
plt.xlabel("Parameter value (ε / α / c / Q0)")
plt.ylabel("Average reward over last 100,000 steps")
plt.title("Nonstationary parameter study (Ex 2.11)")
plt.grid(True, which="both")
plt.legend()
plt.show()
