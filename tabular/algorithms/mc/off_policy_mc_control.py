class EpsilonSoftPolicy: # Behavior policy (policy_b)
    def __init__(self, env, epsilon, rng):
        self.env = env
        self.epsilon = epsilon
        self.rng = rng
        self.pi_table = {}

        for s in env.states():
            actions = list(env.actions(s))
            if not actions:
                self.pi_table[s] = {}
                continue
            
            # set all actions to have p = epsilon / |A(s)|
            p = self.epsilon / len(actions)
            self.pi_table[s] = {a: p for a in actions}

            # Randomly select an initial greedy action
            a_greedy = rng.choice(actions)
            self.pi_table[s][a_greedy] += 1.0 - self.epsilon
    
    def pi(self, s): return self.pi_table[s]

    def update_epsilon_greedy(self, s, greedy_action):
        actions = list(self.env.actions(s))
        if not actions:
            self.pi_table[s] = {}
            return

        base = self.epsilon / len(actions)
        dist = {a: base for a in actions}
        dist[greedy_action] += 1.0 - self.epsilon
        self.pi_table[s] = dist
    
    def policy_sample(self, s, rng):
        actions = list(self.env.actions(s))
        if not actions:
            raise ValueError(f"No actions available in state {s}")
        
        dist = self.pi(s)
        probs = [dist.get(a, 0.0) for a in actions]
        total = sum(probs)
        if total <= 0: probs = [1.0 / len(actions)] * len(actions)
        else: probs = [x / total for x in probs]
        return rng.choice(actions, p=probs)

class DeterministicPolicy: # target policy.(policy_pi)
    def __init__(self, env, rng):
        self.env = env
        self.rng = rng
        self.pi = {}

        for s in self.env.states():
            actions = list(self.env.actions(s))
            self.pi[s] = self.rng.choice(actions) if actions else None

    def action(self, s): return self.pi[s]
    
    def set_greedy(self, s, greedy_actions):
        actions = list(self.env.actions(s))
        if not actions:
            self.pi[s] = None
            return

        a = sorted(greedy_actions)[0]
        self.pi[s] = a

class OffPolicyMCControl:
    def __init__(self, env, policy_pi, policy_b, gamma, tolerance):
        self.env = env
        self.policy_pi = policy_pi
        self.policy_b = policy_b
        self.gamma = gamma
        self.tolerance = tolerance

        self.Q = {s: {a: 0.0 for a in self.env.actions(s)} for s in self.env.states()}
        self.C = {s: {a: 0.0 for a in self.env.actions(s)} for s in self.env.states()}

    def episode(self, rng):
        episode = []

        s0, r0, done = self.env.reset(rng)
        if done: return []
        
        s = s0
        while True:
            a = self.policy_b.policy_sample(s, rng)
            s_prime, r, done = self.env.step(s, a, rng)
            episode.append((s, a, r))

            if done: break
            s = s_prime
        return episode
    
    def run(self, episodes, rng):
        for _ in range(episodes):
            episode = self.episode(rng)

            G = 0.0
            W = 1.0
            states_to_update_b = set()

            for t in range(len(episode)-1, -1, -1):
                s_t, a_t, r_t = episode[t]

                G = self.gamma * G + r_t
                self.C[s_t][a_t] += W
                self.Q[s_t][a_t] += (W / self.C[s_t][a_t]) * (G - self.Q[s_t][a_t])

                legal_actions = list(self.env.actions(s_t))
                if not legal_actions:
                    continue

                max_q = max(self.Q[s_t][a] for a in legal_actions)
                greedy_actions = [a for a in legal_actions
                                if abs(self.Q[s_t][a] - max_q) < self.tolerance]

                self.policy_pi.set_greedy(s_t, greedy_actions)
                states_to_update_b.add(s_t)

                if a_t != self.policy_pi.action(s_t):
                    break

                b_prob = self.policy_b.pi(s_t).get(a_t, 0.0)
                if b_prob == 0.0:
                    raise ValueError("Coverage violated")
                W *= 1.0 / b_prob

            for s in states_to_update_b:
                a_star = self.policy_pi.action(s)
                if a_star is None:
                    continue
                self.policy_b.update_epsilon_greedy(s, a_star)

        
        return self.Q, self.policy_pi