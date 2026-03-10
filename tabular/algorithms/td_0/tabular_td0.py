class TestPolicy: # equiprobability policy(arbitrary)
    def __init__(self, env):
        self.env = env
        self.pi_table = {}

        for s in self.env.states():
            actions = list(self.env.actions(s))
            if not actions:
                self.pi_table[s] = {}
                continue

            p = 1 / len(actions)
            self.pi_table[s] = {a: p for a in actions}

    def pi(self, s): return self.pi_table[s]
    
    def policy_sample(self, s, rng):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions available in state {s}")
        
        dist = self.pi(s)
        probs = [dist.get(a, 0.0) for a in actions]
        total = sum(probs)
        if total <= 0: probs = [1.0 / len(actions)] * len(actions)
        else: probs = [p / total for p in probs]
        return rng.choice(actions, p=probs)

class TD0:
    def __init__(self, env, gamma, alpha):
        self.env = env
        self.policy = TestPolicy(self.env)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.V = {s: 0 for s in self.env.states()}
        if not 0 < self.alpha <= 1: raise ValueError(f"Invalid alpha parameter. Alpha = {self.alpha}")
    
    def run(self, episodes, rng):
        for _ in range(episodes):
            s_t = self.env.reset()
            done = False
            while not done:
                a_t = self.policy.policy_sample(s_t, rng)
                s_prime, r, done, _ = self.env.step(a_t, rng) # where step returns next state, reward, terminating flag, and optional info

                self.V[s_t] += self.alpha * (r + self.gamma * self.V[s_prime] - self.V[s_t])
                s_t = s_prime
        
        return self.V

# I think I should no longer pass rng into env/ policy. They should instead be defined in the env and policy. reduces variables to pass.


