class TargetPolicy:  # Greedy
    def __init__(self, env):
        self.env = env

    def _greedy_actions(self, s, actions, Q):
        max_q = max(Q.get(s, {}).get(act, 0.0) for act in actions)
        return [act for act in actions if Q.get(s, {}).get(act, 0.0) == max_q]

    def prob(self, s, a, Q):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions available for state: {s}")
        if a not in actions: return 0.0

        greedy_actions = self._greedy_actions(s, actions, Q)
        return (1.0 / len(greedy_actions)) if a in greedy_actions else 0.0

class BehaviorPolicy:  # Epsilon-greedy
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = float(epsilon)

    def _greedy_actions(self, s, actions, Q):
        max_q = max(Q.get(s, {}).get(act, 0.0) for act in actions)
        return [act for act in actions if Q.get(s, {}).get(act, 0.0) == max_q]

    def prob(self, s, a, Q):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions available for state: {s}")
        if a not in actions: return 0.0

        base = self.epsilon / len(actions)
        greedy_actions = self._greedy_actions(s, actions, Q)

        if a in greedy_actions:
            return base + (1.0 - self.epsilon) / len(greedy_actions)
        return base

    def sample(self, rng, s, Q):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions available for state: {s}")

        if rng.random() < self.epsilon:
            return actions[rng.integers(len(actions))]

        greedy_actions = self._greedy_actions(s, actions, Q)
        return greedy_actions[rng.integers(len(greedy_actions))]

class OffPolicyNStepSarsa:
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

        self.b = BehaviorPolicy(self.env, self.epsilon)
        self.pi = TargetPolicy(self.env)
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
            RHO = [1.0] * buf_len

            s, reset_info = self.env.reset(self.rng)
            a = self.b.sample(self.rng, s, self.Q)
            S[t] = s
            A[t] = a

            while True:
                if t < T:
                    s_prime, r, terminated, truncated, step_info = self.env.step(a)
                    done = terminated or truncated
                    S[(t + 1) % buf_len] = s_prime
                    R[(t + 1) % buf_len] = r

                    if done : T = t + 1
                    else:
                        a_prime = self.b.sample(self.rng, s_prime, self.Q)
                        A[(t + 1) % buf_len] = a_prime
                        a = a_prime
                        RHO[(t + 1) % buf_len] = self.pi.prob(S[(t + 1) % buf_len], A[(t + 1) % buf_len], self.Q) / self.b.prob(S[(t + 1) % buf_len], A[(t + 1) % buf_len], self.Q)
                    s = s_prime

                tau = t - self.n + 1
                if tau >= 0:
                    rho = 1.0
                    G = 0.0
                    t_end = tau + self.n if T == INF else min(tau + self.n, T)

                    for i in range(tau + 1, t_end + 1):
                        rho *= RHO[i % buf_len]
                        G += self.gamma_pows[i - tau - 1] * R[i % buf_len]

                    if tau + self.n < T: 
                        G += self.gamma_pows[self.n] * self.q(S[(tau + self.n) % buf_len], A[(tau + self.n) % buf_len])
                    
                    q_sa = self.q(S[tau % buf_len], A[tau % buf_len])
                    self.Q[S[tau % buf_len]][A[tau % buf_len]] += self.alpha * rho * (G - q_sa)
                        # make pi greedy
                if tau == T - 1: break
                t += 1
        
        return self.pi

