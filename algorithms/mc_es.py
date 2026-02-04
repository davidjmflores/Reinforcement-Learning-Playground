# Monte Carlo ES (Exploring Starts)

class MCES:
    def __init__(self, env, policy, gamma):
        self.env = env
        self._states = list(env.states())
        self.policy = policy # Start w/ policy that assigns all A(s) w/ equal probability(or arbitrarilly)
        self.Q = {s: {a: 0.0 for a in env.actions(s)} for s in self._states} # Q(s,a) (arbitrary), for all s in S, a in A(s)
        self.N = {s: {a: 0 for a in env.actions(s)} for s in self._states} # For incremental updating
        self.tolerance = 1e-3
        self.gamma = gamma

    def policy_sample(self, s, rng):
        actions = sorted(self.env.actions(s))
        if not actions:
            raise ValueError(f"No actions available in state {s}")
        dist = self.policy.pi(s)
        probs = [dist.get(a, 0.0) for a in actions]
        total = sum(probs)
        if total <= 0:
            probs = [1.0 / len(actions)] * len(actions)
        else:
            probs = [p / total for p in probs]
        return int(rng.choice(actions, p=probs))

    
    def episode(self, s0, a0, rng):
        episode = []
        s = s0
        a = a0

        while True:
            s_prime, r, done = self.env.step(s, a, rng)
            episode.append((s, a, r))
            if done:
                break
            s = s_prime
            a = self.policy_sample(s, rng)
        return episode
    
    def exploring_starts(self, episodes, rng):
        for _ in range(episodes):
            if hasattr(self.env, "exploring_start"):
                s0, a0 = self.env.exploring_start(rng)
            else:
                s0 = rng.choice(self._states)
                actions0 = list(self.env.actions(s0))
                if not actions0:
                    continue
                a0 = rng.choice(actions0)

            episode = self.episode(s0, a0, rng)
            T = len(episode)
        
            # Backward pass:
            G = 0.0
            Gs = [0.0] * T
            for t in range(T - 1, -1, -1):
                r = episode[t][2]
                G = self.gamma * G + r
                Gs[t] = G
            
            # Forward pass
            seen = set()
            for t in range(T):
                s_t = episode[t][0]
                a_t = episode[t][1]
                r_t = episode[t][2]

                q_sa = (s_t, a_t)
                if q_sa in seen:
                    continue
                seen.add(q_sa)

                # Q(st, At) = average(Returns(St, At))
                self.N[s_t][a_t] += 1 # When state-action pair selected, increment selection-counter for it
                step_size = 1.0 / self.N[s_t][a_t] # Increment step size 
                self.Q[s_t][a_t] += step_size * (Gs[t] - self.Q[s_t][a_t])

                # pi(St) = argmax over a of Q(St, a)
                legal_actions = list(self.env.actions(s_t))
                if not legal_actions: continue
                max_q = max(self.Q[s_t][a] for a in legal_actions)
                greedy_actions = [a for a in legal_actions
                                if abs(self.Q[s_t][a] - max_q) < self.tolerance]
                self.policy.set_greedy(s_t, greedy_actions)

        return self.Q, self.policy



    