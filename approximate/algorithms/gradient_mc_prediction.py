import numpy as np

class Equiprobable:
    def __init__(self, env): self.env = env

    def sample(self, rng, s):
        actions = list(self.env.actions(s))
        if not actions: raise(ValueError(f"No actions for state {s}"))
        return actions[rng.integers(len(actions))]

class LinearValueFunction:
    def __init__(self, feature_fn, d):
        self.feature_fn = feature_fn
        self.d = d
    
    def __call__(self, s, w):
        x = self.feature_fn(s)
        return np.dot(w, x)
    
    def grad(self, s, w):
        return self.feature_fn(s)
        
class GradientMCPrediction:
    def __init__(self, rng, env, alpha, value_fn, d, policy, gamma = 1.0):
        if alpha <= 0: raise ValueError("alpha must be > 0")
        if d <= 0: raise ValueError("d must be > 0")

        self.rng = rng
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.value_fn = value_fn
        self.policy = policy
        self.w = np.zeros(d, dtype=float)
    
    def generate_episode(self):
        states = []
        rewards = []

        s_t, _ = self.env.reset()
        states.append(s_t)

        terminated = truncated = False
        while not (terminated or truncated):
            a_t = self.policy.sample(self.rng, s_t)
            s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
            rewards.append(r_tp1)
            states.append(s_tp1)
            s_t = s_tp1

        return states, rewards
    
    def returns_from_episode(self, rewards):
        T = len(rewards)
        G = np.zeros(T, dtype=float)
        g = 0.0
        for t in reversed(range(T)):
            g = rewards[t] + self.gamma * g
            G[t] = g
        return G
    
    def run_episode(self):
        states, rewards = self.generate_episode()
        returns = self.returns_from_episode(rewards)

        T = len(rewards)
        for t in range(T):
            s_t = states[t]
            grad = self.value_fn.grad(s_t, self.w)
            v = self.value_fn(s_t, self.w)
            self.w += self.alpha * (returns[t] - v) * grad

        return self.w.copy()
    
    def run(self, episodes):
        for _ in range(episodes):
            self.run_episode()
        return self.w.copy()
    




    