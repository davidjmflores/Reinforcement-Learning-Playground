import numpy as np

class FirstVisitMCPrediction:
    def __init__(self, env, policy, v0, gamma):
        self.policy = policy
        self.env = env
        self.v = {s: v0 for s in self.env.states()}
        self.returns_sum = {s: 0 for s in self.env.states()}
        self.returns_count = {s: 0 for s in self.env.states()}
        self.gamma = gamma

    def policy_sample(self, s, rng):
        return int(rng.choice(self.env.actions(s), p=self.policy.pi(s)))

    def episode(self, rng):
        episode = []
        s, r, done = self.env.reset(rng)
        while not done:
            a = self.policy_sample(s, rng)
            s_prime, r, done = self.env.step(s, a, rng)
            episode.append((s, a, r))
            s = s_prime
        return episode

    def prediction(self, episodes, rng):
        for _ in range(episodes):
            episode = self.episode(rng)
            T = len(episode)

            # Backward pass:
            G = 0.0
            Gs = [0.0] * T                       # Gs[t] = return starting at time t
            for t in range(T - 1, -1, -1):
                r = episode[t][2]
                G = self.gamma * G + r
                Gs[t] = G

            # 2) Forward pass: first-visit updates
            seen = set()
            for t in range(T):
                s_t = episode[t][0]
                if s_t in seen:
                    continue
                seen.add(s_t)

                self.returns_sum[s_t] += Gs[t]
                self.returns_count[s_t] += 1
                self.v[s_t] = self.returns_sum[s_t] / self.returns_count[s_t]

        return self.v


