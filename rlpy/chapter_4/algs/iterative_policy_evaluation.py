# gamma: discount factor
# theta: small threshold determining accuracy of estimation
# v_0: value init
class IterativePolicyEvaluation:
    def __init__(self, env, policy, v0, theta, gamma):
        self.env = env
        self.policy = policy
        self.theta = theta
        self.gamma = gamma
        self.v = {s: v0 for s in self.env.states()}

    def iterate(self, record=False): # In-place updates/ sweeps
        history = [] # Added so that I can visualize policy eval like on page 77 
        k = 0

        if record: history.append(self.v.copy()) # V_0

        while True:
            delta = 0.0
            for s in self.env.states():
                v_old = self.v[s]
                v_new = 0.0
                pi = self.policy.pi(s)

                for a in self.env.actions:
                    transitions = self.env.transitions(s, a)
                    for (p, s_prime, r) in transitions:
                        v_new += pi[a] * p * (r + self.gamma * self.v[s_prime])
                
                delta = max(delta, abs(v_old - v_new))
                self.v[s] = v_new
            
            k += 1
            if record: history.append(self.v.copy()) # V_k after sweep k

            if delta < self.theta: break

        return (self.v, history) if record else self.v



