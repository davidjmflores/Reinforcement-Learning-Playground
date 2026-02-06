# On-policy first-visit MC control (for epsilon-soft policies), estimates pi = pi*

# Policy used for policy improvement in PI, VI, and MCES

class EpsilonSoftPolicy:
    def __init__(self, env, epsilon, rng):
        self.env = env
        self.epsilon = epsilon
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
            a_greedy = rng.choice(list(actions))
            self.pi_table[s][a_greedy] += 1.0 - self.epsilon

    def pi(self, s):
        return self.pi_table[s]

    def set_epsilon_greedy(self, s, greedy_actions):
        if not greedy_actions:
            raise ValueError(f"set_greedy called with empty greedy_actions for state {s}")

        actions = list(self.env.actions(s))
        if not actions:
            self.pi_table[s] = {}
            return

        # assign prob to non-greedy actions as (epsilon) / |actions(s)|
        p = self.epsilon / len(actions)
        self.pi_table[s] = {a: p for a in actions}

        # assign prob to greedy actions as (1 - epsilon) / |greedy_actions(s)|
        p = (1 - self.epsilon) / len(greedy_actions)
        for a in greedy_actions:
            if a not in self.pi_table[s]:
                raise ValueError(f"Greedy action {a} not legal in state {s}")
            self.pi_table[s][a] += p
            
class OnPolicyFVMC:
    def __init__(self, env, epsilon, gamma, tolerance, rng):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.tolerance = tolerance
        self._states = self.env.states()

        self.policy = EpsilonSoftPolicy(self.env, epsilon, rng)
        self.Q = {s: {a: 0.0 for a in self.env.actions(s)} for s in self._states}
        self.N = {s: {a: 0 for a in self.env.actions(s)} for s in self._states}

    # Takes a sample action from pi(a|s)
    def policy_sample(self, s, rng):
        actions = list(self.env.actions(s))
        if not actions:
            raise ValueError(f"No actions available in state {s}")
        
        dist = self.policy.pi(s)
        probs = [dist.get(a, 0.0) for a in actions]
        total = sum(probs)
        if total <= 0: probs = [1.0 / len(actions)] * len(actions)
        else: probs = [p / total for p in probs]
        return rng.choice(actions, p=probs)
    
    def episode(self, rng):
        episode = []

        s0, r0, done = self.env.reset(rng)
        
        if done:
            episode.append((s0, None, r0))
            return episode
        
        s = s0
        a = self.policy_sample(s, rng)

        while True:
            s_prime, r, done = self.env.step(s, a, rng)
            episode.append((s, a, r))

            if done: break

            s = s_prime
            a = self.policy_sample(s, rng)
        return episode
    
    def run(self, episodes, rng):
        for _ in range(episodes):
            episode = self.episode(rng)
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
                
                self.policy.set_epsilon_greedy(s_t, greedy_actions)
        
        return self.Q, self.policy


    
    

    



