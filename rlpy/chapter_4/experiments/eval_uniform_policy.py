from envs.gridworld_terminating import GridworldTerminating
from algs.iterative_policy_evaluation import IterativePolicyEvaluation
from policies.uniform_random import UniformRandom

gamma = 1.0

env = GridworldTerminating()
policy = UniformRandom(env.actions)
evaluator = IterativePolicyEvaluation(env, policy, v0=0.0, theta=1e-4, gamma=gamma)
V = evaluator.iterate()
