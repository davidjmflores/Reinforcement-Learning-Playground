class GreedyPolicy: # Target Policy
    def __init__(self, env):
        self.env = env

    def prob(self, s, a, Q):
        return 0
    
    def make_greedy(self, s, Q):
        return 0
    
class EpsilonGreedyPolicy: # Behavioral Policy
   def __init__(self, env, epsilon):
      self.env = env
      self.epsilon = epsilon

   def sample(self, rng, s, Q):
      actions = list(self.env.actions(s))
      if not actions: raise ValueError(f"No actions available in state {s}")

      if rng.random() < self.epsilon: return actions[rng.integers(len(actions))]
        
      # exploit
      q_vals = [Q.get(s, {}).get(a, 0.0) for a in actions]
      max_q = max(q_vals)
      greedy_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
      return greedy_actions[rng.integers(len(greedy_actions))]
    
class NStepTreeBackup:
    def __init__(self, rng, env, epsilon, gamma, alpha, n):
        self.rng = rng
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.n = n

        self.pi = GreedyPolicy(self.env)
        self.b = EpsilonGreedyPolicy(self.env, self.epsilon)
        self.Q = {}
    
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

            s, reset_info = self.env.reset(self.rng)
            a = self.b.sample(self.rng, s)
            S[t % buf_len] = s
            A[t % buf_len] = a

            while True:
                if t < T:
                    s_prime, r, terminated, truncated, step_info = self.env.step(a)
                    done = terminated or truncated
                    S[(t + 1) % buf_len] = s_prime
                    R[(t + 1) % buf_len] = r

                    if not done:
                        a_prime = self.b.sample(s_prime)
                        A[(t + 1) % buf_len] = a_prime
                        a = a_prime
                    s = s_prime
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0.0

                    # will resume tmrw


