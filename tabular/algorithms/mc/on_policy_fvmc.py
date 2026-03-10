# On-policy first-visit MC control (for epsilon-soft policies), estimates pi = pi*

# Policy used for policy improvement in PI, VI, and MCES

class EpsilonSoftPolicy:
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon
        self.pi_table = {}

    def _ensure_state(self, s):
        if s in self.pi_table:
            return
        actions = list(self.env.actions(s))
        self.pi_table[s] = {a: 1.0/len(actions) for a in actions} if actions else {}

    def sample(self, rng, s):
        self._ensure_state(s)
        dist = self.pi_table[s]
        actions = list(dist.keys())
        probs = [dist[a] for a in actions]
        idx = rng.choice(len(actions), p=probs)
        return actions[idx]

    def make_epsilon_greedy(self, s, greedy_actions):
        self._ensure_state(s)
        actions = list(self.env.actions(s))
        if not actions:
            self.pi_table[s] = {}
            return
        if not greedy_actions:
            raise ValueError(f"empty greedy_actions for state {s}")

        base = self.epsilon / len(actions)
        dist = {a: base for a in actions}
        bonus = (1.0 - self.epsilon) / len(greedy_actions)
        for a in greedy_actions:
            dist[a] += bonus
        self.pi_table[s] = dist

class OnPolicyFVMC:
    def __init__(self, env, epsilon, gamma, tolerance, rng):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.tolerance = tolerance
        self._states = self.env.states()
        self.total_steps = 0
        self.episode_end_steps = []
        self.rng = rng

        self.policy = EpsilonSoftPolicy(self.env, epsilon)
        self.Q = {s: {a: 0.0 for a in self.env.actions(s)} for s in self._states}
        self.N = {s: {a: 0 for a in self.env.actions(s)} for s in self._states}
    
    def episode(self):
        episode = []

        s0, _ = self.env.reset()
        
        s = s0
        a = self.policy.sample(self.rng, s)

        while True:
            s_prime, r, terminated, truncated, _ = self.env.step(a)
            done = terminated or truncated
            episode.append((s, a, r))

            self.total_steps += 1
            if done: 
                self.episode_end_steps.append(self.total_steps)
                break

            s = s_prime
            a = self.policy.sample(self.rng, s)
        return episode
    
    def run(self, episodes):
        for _ in range(episodes):
            episode = self.episode()
            T = len(episode)

            G = 0.0
            Gs = [0.0] * T

            # backward pass
            for t in range(T - 1, -1, -1):
                r = episode[t][2]
                G = self.gamma * G + r
                Gs[t] = G
            
            seen = set()
            for t in range(T):
                s_t = episode[t][0]
                a_t = episode[t][1]
                if a_t is None: continue


                q_sa = (s_t, a_t)
                if q_sa in seen:
                    continue
                seen.add(q_sa)

                # incremental return update
                self.N[s_t][a_t] += 1
                step_size = 1.0 / self.N[s_t][a_t]
                self.Q[s_t][a_t] += step_size * (Gs[t] - self.Q[s_t][a_t])

                legal_actions = list(self.env.actions(s_t))
                if not legal_actions: continue

                max_q = max(self.Q[s_t][a] for a in legal_actions)
                greedy_actions = [a for a in legal_actions
                                  if abs(self.Q[s_t][a] - max_q) < self.tolerance]
                
                self.policy.make_epsilon_greedy(s_t, greedy_actions)
        
        return self.Q, self.policy, self.episode_end_steps
