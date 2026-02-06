from environments.ch4_jacks_car_rental import JacksCarRental
from algorithms.dp.policy_iteration import PolicyIteration
from policies.tabular_stochastic_policy import JacksTabularStochasticPolicy
from experiments.chapter_4.plot_eval_snapshots import plot_jacks_policy
env = JacksCarRental()
policy = JacksTabularStochasticPolicy(env)

pi = PolicyIteration(
    env,
    policy,
    v0=0.0,
    theta=1e-4,
    gamma=0.9
)

policy_final, V_final, log = pi.iterate(record=True)

# plot a few snapshots
plot_jacks_policy(log["policies_by_iter"][0], env, title="Policy after improvement 0")
plot_jacks_policy(log["policies_by_iter"][1], env, title="Policy after improvement 1")
plot_jacks_policy(log["policies_by_iter"][-1], env, title="Final policy")


