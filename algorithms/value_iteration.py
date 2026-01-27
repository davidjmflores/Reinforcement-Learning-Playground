import math

class ValueIteration:
    def __init__(self, env, policy, v0, theta, gamma):
        self.env = env
        self.policy = policy
        self.theta = theta
        self.gamma = gamma
        self.v = {s: v0 for s in self.env.states()}
        self.tolerance = 1e-6
    
    def iterate(self, record=False):
        log = None
        if record:
            log = {"values_by_iter": []}
        
        while True:
            if record: log["values_by_iter"].append(self.v.copy())

            delta = self.evaluation_sweep()

            if delta < self.theta: break

        self.greedy_policy_from_V()
        if record: log["values_by_iter"].append(self.v.copy())

        return (self.policy, self.v, log) if record else (self.policy, self.v)

    def evaluation_sweep(self):
        
        delta = 0.0
        for s in self.env.states():
            v_old = self.v[s]
            v_new = - math.inf
            actions = self.env.actions(s)

            if not actions: v_new = 0.0
            else:
                for a in actions:
                    q_sa = 0.0
                    for (p, s_prime, r) in self.env.transitions(s, a):
                        q_sa += p * (r + self.gamma * self.v[s_prime])
                    v_new = max(q_sa, v_new)
            self.v[s] = v_new
            delta = max(delta, abs(v_old - v_new))
        
        return delta

    def greedy_policy_from_V(self):
        for s in self.env.states():
            actions = self.env.actions(s)
            if not actions: continue
            q = {}
            for a in actions:
                q_sa = 0.0
                for (p, s_prime, r) in self.env.transitions(s, a):
                    q_sa += p * (r + self.gamma * self.v[s_prime])
                q[a] = q_sa
            
            max_q = max(q.values())
            greedy_actions = [a for a, val in q.items()
                              if abs(val - max_q) < self.tolerance]
            
            self.policy.set_greedy(s, greedy_actions)
