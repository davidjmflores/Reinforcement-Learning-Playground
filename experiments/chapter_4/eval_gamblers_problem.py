from environments.ch4_gamblers_problem import GamblersProblem
from algorithms.value_iteration import ValueIteration
from policies.tabular_stochastic_policy import GamblersTabularPolicy
from experiments.chapter_4.plot_eval_snapshots import plot_value_sweeps, plot_final_policy

env = GamblersProblem()
policy = GamblersTabularPolicy(env)

pi = ValueIteration(
    env,
    policy,
    v0 = 0.0,
    theta = 1e-4,
    gamma = 1.0,
)

policy_final, V, log = pi.iterate(record=True)

plot_value_sweeps(log, env, sweeps=[0,1,2,len(log["values_by_iter"])-1])
plot_final_policy(policy_final, env)
