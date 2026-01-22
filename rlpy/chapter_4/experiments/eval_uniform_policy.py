from envs.gridworld_terminating import GridworldTerminating
from algs.iterative_policy_evaluation import IterativePolicyEvaluation
from policies.uniform_random import UniformRandom
from experiments.plot_eval_snapshots import plot_selected_ks

gamma = 1.0

env = GridworldTerminating()
policy = UniformRandom(env.actions)
evaluator = IterativePolicyEvaluation(env, policy, v0=0.0, theta=1e-4, gamma=gamma)

V, history = evaluator.iterate(record=True)
plot_selected_ks(history, ks=[0, 1, 2, 3, 10, len(history)-1], n_rows=env.n_rows, n_cols=env.n_cols)
