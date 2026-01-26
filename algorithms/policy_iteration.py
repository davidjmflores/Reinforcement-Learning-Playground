# gamma: discount factor
# theta: small threshold determining accuracy of estimation
# v_0: value init

# the deterministic pseudocode presented on pg. 80 isn't applicable to Example 4.1
class PolicyIteration: 
    def __init__(self, env, policy, v0, theta, gamma):
        self.env = env
        self.policy = policy
        self.theta = theta
        self.gamma = gamma
        self.v = {s: v0 for s in self.env.states()}
        self.tolerance = 1e-6 # for argmax over actions. Accounts for minimal floating point issues

    # Added so that same file can be used for both state-indep and state-dep actions
    def actions_for(self, s):
        # If env exposes actions(s), use it (Jack's).
        if hasattr(self.env, "actions") and callable(self.env.actions):
            return self.env.actions(s)

        # Otherwise assume env.actions is an iterable (Gridworld).
        return self.env.actions

    def iterate(self, record=False, record_eval=False): # For running the solver
        log = None
        if record:
            log = {
                "values_by_iter": [],     # V after each policy evaluation
                "policies_by_iter": [],   # policy snapshot after each improvement
                "eval_histories": []      # list of histories; each history is [V_0, V_1, ...]
            }

        while True:
            # capture the return from policy_evaluation
            if record_eval:
                V_eval, eval_hist = self.policy_evaluation(record=True)
            else:
                V_eval = self.policy_evaluation(record=False)
                eval_hist = None

            if record:
                log["values_by_iter"].append(V_eval.copy())
                if record_eval:
                    log["eval_histories"].append(eval_hist)

            stable = self.policy_improvement()

            if record:
                log["policies_by_iter"].append(self.policy.snapshot())

            if stable:
                return (self.policy, self.v, log) if record else (self.policy, self.v)

    def policy_evaluation(self, record=False):  
        if record:
            history = []
            history.append(self.v.copy())

        while True:
            delta = 0.0
            for s in self.env.states():
                v_old = self.v[s]
                v_new = 0.0
                pi = self.policy.pi(s)

                for a in self.actions_for(s):
                    for (p, s_prime, r) in self.env.transitions(s, a):
                        v_new += pi.get(a, 0.0) * p * (r + self.gamma * self.v[s_prime])

                delta = max(delta, abs(v_old - v_new))
                self.v[s] = v_new

            if record:
                history.append(self.v.copy())

            if delta < self.theta:
                break

        return (self.v, history) if record else self.v

    def policy_improvement(self):
        policy_stable = True

        for s in self.env.states():
            old_pi = self.policy.pi(s).copy()

            q = {}
            for a in self.actions_for(s):
                q_sa = 0.0
                for (p, s_prime, r) in self.env.transitions(s, a):
                    q_sa += p * (r + self.gamma * self.v[s_prime])
                q[a] = q_sa

            max_q = max(q.values())
            greedy_actions = [a for a, val in q.items()
                              if abs(val - max_q) < self.tolerance]

            self.policy.set_greedy(s, greedy_actions)

            if old_pi != self.policy.pi(s):
                policy_stable = False

        return policy_stable