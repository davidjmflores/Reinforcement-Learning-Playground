class EpsilonGreedyPolicy:
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon

    def sample(self, rng, s, Q):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions available in state: {s}")

        if rng.random() < self.epsilon: return actions[rng.integers(len(actions))]

        q_vals = [Q.get(s, {}).get(a, 0.0) for a in actions]
        max_q = max(q_vals)
        greedy_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
        return greedy_actions[rng.integers(len(greedy_actions))]

class NStepSarsa:
    def __init__(self, rng, env, gamma, alpha, epsilon, n):
        self.rng = rng
        self.env = env

        self.gamma = gamma 
        if not 0 < self.gamma <= 1.0: raise ValueError(f"Invalid parameter value: gamma = {self.gamma}")
        self.alpha = alpha
        if not 0 < self.alpha <= 1.0: raise ValueError(f"Invalid parameter value: alpha = {self.alpha}")
        self.epsilon = epsilon
        if not 0 < self.epsilon <= 1.0: raise ValueError(f"Invalid parameter value: epsilon = {self.epsilon}")
        self.n = int(n)
        if not self.n > 0: raise ValueError(f"Invalid parameter value: n = {self.n}")

        self.gamma_pows = [1.0] * (self.n + 1)
        for k in range(1, self.n + 1):
            self.gamma_pows[k] = self.gamma_pows[k - 1] * self.gamma

        self.policy = EpsilonGreedyPolicy(self.env, self.epsilon)
        self.Q = {}
    
    def q(self, s, a):
        self.Q.setdefault(s, {})
        self.Q[s].setdefault(a, 0.0)
        return self.Q[s][a]
    
    def run(self, episodes):
        # info = []
        for ep in range(episodes):
            INF = float("inf")
            T = INF
            t = 0

            buf_len = self.n + 1
            tau = 0
            S = [None] * buf_len
            A = [None] * buf_len
            R = [0.0] * buf_len

            s, reset_info = self.env.reset(self.rng)
            S[t] = s
            a = self.policy.sample(self.rng, s, self.Q)
            A[t] = a

            while True:
                if t < T:
                    s_prime, r, terminated, truncated, step_info = self.env.step(a)
                    S[(t + 1) % buf_len] = s_prime
                    R[(t + 1) % buf_len] = r
                    done = terminated or truncated

                    if done: T = t + 1
                    else:
                        a_prime = self.policy.sample(self.rng, s_prime, self.Q)
                        A[(t+1) % buf_len] = a_prime
                        a = a_prime
                    s = s_prime
                
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0.0
                    t_end = tau + self.n if T == INF else min(tau + self.n, T)

                    for i in range(tau + 1, t_end + 1):
                        G += self.gamma_pows[i - tau - 1] * R[i % buf_len]

                    if tau + self.n < T: 
                        G += self.gamma_pows[self.n] * self.q(S[(tau + self.n) % buf_len], A[(tau + self.n) % buf_len])
                        
                    q_sa = self.q(S[tau % buf_len], A[tau % buf_len])
                    self.Q[S[tau % buf_len]][A[tau % buf_len]] += self.alpha * (G - q_sa)
            
                if tau == T - 1: break
                t += 1
        
        return self.Q




