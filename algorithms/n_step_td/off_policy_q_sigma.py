class EpsilonGreedyPolicy: # Behavioral Policy
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon
    
    def prob(self, s, a, Q):
        actions = list(self.env.actions(s))
        if not actions: return 0.0

        q_vals = [Q.get(s, {}).get(a_i, 0.0) for a_i in actions]
        max_q = max(q_vals)
        greedy = [a_i for a_i, q in zip(actions, q_vals) if q == max_q]
        prob_base = self.epsilon / len(actions)
        if not greedy: return prob_base

        return ((1.0 - self.epsilon) / len(greedy)) + prob_base if a in greedy else prob_base
       

    def sample(self, rng, s, Q):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions available in state {s}")

        if rng.random() < self.epsilon: return actions[rng.integers(len(actions))]
            
        q_vals = [Q.get(s, {}).get(a, 0.0) for a in actions]
        max_q = max(q_vals)
        greedy_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
        return greedy_actions[rng.integers(len(greedy_actions))]
   
class OffPolicyQSigma:
    def __init__(self, rng, env, epsilon, gamma, alpha, n, sigma):
        self.rng = rng
        self.env = env

        self.epsilon = float(epsilon)
        if not (0.0 <= self.epsilon <= 1.0): raise ValueError(f"Invalid parameter value: epsilon = {self.epsilon}")
        self.gamma = float(gamma)
        if not (0.0 <= self.gamma <= 1.0): raise ValueError(f"Invalid parameter value: gamma = {self.gamma}")
        self.alpha = float(alpha)
        if not (0.0 < self.alpha <= 1.0): raise ValueError(f"Invalid parameter value: alpha = {self.alpha}")
        self.n = int(n)
        if not (self.n > 0): raise ValueError(f"Invalid parameter value: n = {self.n}")
        self.sigma = float(sigma)
        if not (0.0 <= self.sigma <= 1.0): raise ValueError(f"Invalid parameter value: sigma = {self.sigma}")

        self.b = EpsilonGreedyPolicy(self.env, self.epsilon)
        self.Q = {}
    
    def pi_prob(self, s, a): # implicit target policy
        actions = list(self.env.actions(s))
        if not actions:
            return 0.0

        q_vals = [self.q(s, a_i) for a_i in actions]
        max_q = max(q_vals)
        greedy = [a_i for a_i, q in zip(actions, q_vals) if q == max_q]

        return 1.0 / len(greedy) if a in greedy else 0.0
    
    def q(self, s, a):
        self.Q.setdefault(s, {})
        self.Q[s].setdefault(a, 0.0)
        return self.Q[s][a]

    def run(self, episodes):
        # info = []
        for ep in range(episodes):
            Inf = float("inf")
            T = Inf
            t = 0
            tau = 0

            buf_len = self.n + 1
            S = [None] * buf_len
            A = [None] * buf_len
            R = [0.0] * buf_len
            Rho = [0.0] * buf_len
            Sigma = [0.0] * buf_len

            s, reset_info = self.env.reset(self.rng)
            # info.append(reset_info)
            a = self.b.sample(self.rng, s, self.Q)
            S[t % buf_len] = s
            A[t % buf_len] = a

            while True:
                if t < T: 
                    s_prime, r, terminated, truncated, step_info = self.env.step(a)
                    # info.append(step_info)
                    S[(t + 1) % buf_len] = s_prime
                    R[(t + 1) % buf_len] = r

                    done = terminated or truncated
                    if done: T = t + 1
                    else:
                        a_prime = self.b.sample(self.rng, s_prime, self.Q)
                        A[(t + 1) % buf_len] = a_prime
                        Sigma[(t + 1) % buf_len] = self.sigma

                        Rho[(t + 1) % buf_len] = self.pi_prob(s_prime, a_prime) / self.b.prob(s_prime, a_prime, self.Q)
                        s = s_prime
                        a = a_prime
                tau = t - self.n + 1
                
                if tau >= 0:
                    G = 0.0
                    if t + 1 < T: G = self.q(S[(t + 1) % buf_len], A[(t + 1) % buf_len])

                    for k in range(min(t + 1, T), tau + 1, -1):
                        v_bar = 0.0
                        if k == T: G = R[T % buf_len]
                        else: 
                            actions = list(self.env.actions(S[k % buf_len]))
                            for a_i in actions: v_bar += self.pi_prob(S[k % buf_len], a_i) * self.q(S[k % buf_len], a_i)
                            G = R[k % buf_len] + self.gamma * (
                                Sigma[k % buf_len] * Rho[k % buf_len]
                                + (1.0 - Sigma[k % buf_len]) * self.pi_prob(S[k % buf_len], A[k % buf_len])
                            ) * (G - self.q(S[k % buf_len], A[k % buf_len])) + self.gamma * v_bar
                    q_sa = self.q(S[tau % buf_len], A[tau % buf_len])
                    self.Q[S[tau % buf_len]][A[tau % buf_len]] += self.alpha * (G - q_sa)
                if tau == T - 1: break
                t += 1
        return self.Q
                        

