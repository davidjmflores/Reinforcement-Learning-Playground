class UniformRandomPolicy:
    def __init__(self, env):
        self.env = env
        self.action_cache = {}

    def sample(self, rng, s_t):
        actions = self.action_cache.get(s_t)

        if actions is None:
            actions = tuple(self.env.actions(s_t))
            if not actions: raise ValueError(f"No actions available in state {s_t}")
            self.action_cache[s_t] = actions

        return actions[rng.integers(len(actions))]
    
class NStepTD:
    def __init__(self, rng, env, gamma, alpha, n):
        self.rng = rng
        self.env = env
        self.gamma = gamma
        if not (0 < self.gamma <= 1.0): raise ValueError(f"Invalid parameter value: gamma = {self.gamma}")
        self.alpha = alpha
        if not (0 < self.alpha <= 1.0): raise ValueError(f"Invalid parameter value: alpha = {self.alpha}")
        self.n = int(n)
        if not self.n > 0: raise ValueError(f"Invalid parameter value: n = {self.n}")

        self.gamma_pows = [1.0] * (self.n + 1)
        for k in range(1, self.n + 1):
            self.gamma_pows[k] = self.gamma_pows[k - 1] * self.gamma
        
        self.policy = UniformRandomPolicy(self.env)
        self.V = {}

    def run(self, episodes):
        # info = []
        for ep in range(episodes):
            INF = float("inf")
            T = INF
            t = 0

            buf_len = self.n + 1
            S = [None] * buf_len
            R = [0.0] * buf_len

            s_t, reset_info = self.env.reset()
            S[0] = s_t
            # info.append(reset_info)

            while True:
                if t < T:
                    a_t = self.policy.sample(self.rng, s_t)
                    s_prime, r, terminated, truncated, step_info = self.env.step(a_t)
                    S[(t + 1) % buf_len] = s_prime
                    R[(t + 1) % buf_len] = r
                    done = terminated or truncated
                    # info.append(step_info)
                    s_t = s_prime

                    if done: T = t + 1
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0.0
                    t_end = tau + self.n if T == INF else min(tau + self.n, T)

                    s_tau = S[tau % buf_len]

                    for i in range(tau + 1, t_end + 1): 
                        G += self.gamma_pows[i - tau - 1] * R[i % buf_len]

                    if tau + self.n < T: 
                        s_boot = S[(tau + self.n) % buf_len]
                        G += self.gamma_pows[self.n] * self.V.get(s_boot, 0.0)

                    v_tau = self.V.get(s_tau, 0.0)
                    self.V[s_tau] = v_tau + self.alpha * (G - v_tau)

                if tau == T - 1: break
                t += 1
        
        return self.V
                    





