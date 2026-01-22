# gamma: discount factor
# theta: small threshold determining accuracy of estimation
# v_0: INIT
class IterativePolicyEvaluation:
    def __init__(self, env, policy, v0, theta, gamma):
        self.policy = policy
        self.theta = theta
        self.env = env
        self.v = {s: v0 for s in self.env.states()}
        self.gamma = gamma

    def iterate(self): # In-place updates/ sweeps
        while True:
            delta = 0.0
            for s in self.env.states():
                v_temp = self.v[s]
                v_new = 0.0
                pi = self.policy.pi(s)
                for a in self.env.actions:
                    transitions = self.env.transitions(s, a)
                    for (p, s_prime, r) in transitions:
                        v_new += pi[a] * p * (r + self.gamma * self.v[s_prime])
                delta = max(delta, abs(v_temp - v_new))
                self.v[s] = v_new
            if delta < self.theta: break



