from environments.ch4_jacks_car_rental import JacksCarRental
from algorithms.policy_iteration import PolicyIteration
from policies.tabular_stochastic_policy import JacksTabularStochasticPolicy

env = JacksCarRental()
policy = JacksTabularStochasticPolicy(env)

pi = PolicyIteration(env, policy, v0=0.0, theta=1e-4, gamma=0.9)

V = pi.policy_evaluation(record=False)

