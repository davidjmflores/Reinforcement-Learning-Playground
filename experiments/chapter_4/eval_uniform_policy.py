from environments.ch4_gridworld_terminating import GridworldTerminating
from algorithms.dp.iterative_policy_evaluation import IterativePolicyEvaluation
from policies.uniform_random import UniformRandom
from experiments.chapter_4.plot_eval_snapshots import plot_selected_ks

gamma = 1.0
v0 = 0.0
theta = 1e-4

env = GridworldTerminating()
policy = UniformRandom(env.actions)
solver = IterativePolicyEvaluation(env, policy, v0=v0, theta=theta, gamma=gamma)

pi_star, V_star, log = solver.iterate(record=True, record_eval=False)
