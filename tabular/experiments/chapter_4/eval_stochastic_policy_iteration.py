from environments.ch4_gridworld_terminating import GridworldTerminating
from algorithms.dp.policy_iteration import PolicyIteration
from policies.tabular_stochastic_policy import TabularStochasticPolicy
from experiments.chapter_4.plot_eval_snapshots import plot_values_by_policy_iteration, plot_eval_sweeps_for_policy, plot_policy

gamma = 1.0

env = GridworldTerminating()
policy = TabularStochasticPolicy(env.states(), env.actions)
evaluator = PolicyIteration(env, policy, v0 =0.0, theta = 1e-4, gamma=gamma)

p, v, log = evaluator.iterate(record=True, record_eval=True)
plot_values_by_policy_iteration(log, env)

plot_eval_sweeps_for_policy(
    log,
    env,
    policy_iter=0,
    ks=[0,1,2,5,10]
)

plot_policy( # Trash
    log["policies_by_iter"][0],
    env,
    title="Initial equiprobable policy"
)

plot_policy( # Also trash. Some scaling issues with plot and arrows
    log["policies_by_iter"][-1],
    env,
    title="Final greedy policy"
)
