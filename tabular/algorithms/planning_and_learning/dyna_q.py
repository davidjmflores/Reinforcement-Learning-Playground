'''
Init Q(s, a) and Model(s, a) for all s in S and a in A(s)
Loop forever:
    a. S gets current non-terminal state
    b. A gets eps-greedy(S, Q)
    c. Take action A; observe resultant R and S'
    d. Q(s, a) += alpha * [R + gamma * max of a of Q(s', a) - Q(s, a)]
    e. model(s, a) gets R and s' (assuming deterministic env)
    f. Loop repeat n times:
        s gets random previously observed state
        a gets random action previously taken in s
        r, s' get model(s, a)
        Q(s, a) += alpha * [r + gamma * max over a of Q(s', a) - Q(s, a)]

'''

class EpsilonGreedy:
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon

    def sample(self, rng, s, Q):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions for state: {s}")

        if rng.random() < self.epsilon: return actions[rng.integers(len(actions))]

        q_vals = [Q.setdefault(s, {}).get(a, 0.0) for a in actions]
        q_max = max(q_vals)
        greedy_actions = [q for a, q in zip(actions, q_vals) if q == q_max]
        return actions[rng.integers(len(greedy_actions))]

class DynaQ:
    def __init__(self, rng, env, epsilon, gamma, alpha, n):
        self.rng = rng
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma 
        self.alpha = alpha
        self.n = n # represents iterations of planning per step (not t-steps)

        self.Q = {}
        self.model = {}
        self.b = EpsilonGreedy(self.env, self.epsilon)
    
    def q_max(self, s, Q):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions for state: {s}")

        q_vals = [Q.setdefault(s, {}).get(a, 0.0) for a in actions]
        q_max = max(q_vals)
        greedy_actions = [q for a, q in zip(actions, q_vals) if q == q_max]
        selected_action = greedy_actions[self.rng.integers(len(greedy_actions))]
        return self.q(s, selected_action)

    def q(self, s, a):
        self.Q.setdefault(s, {})
        self.Q[s].setdefault(a, 0.0)
        return self.Q[s][a]
    
    def model(self, s, a, r, s_prime):
        self.model.setdefault(s, {})
        self.model[s].setdefault(a, {})   
        self.model[s][a].setdefault((r, s_prime), (None, None))
        return self.model[s][a]
     
    def run(self, epsiodes):
        # info = []
        for ep in epsiodes:
            s_t, reset_info = self.env.reset(self.rng)
            # info.append(reset_info)
            while True:
                a_t = self.b.sample(self.rng, s_t, self.Q)

                s_tp1, r_tp1, terminated, truncated, step_info = self.env.step(s_t, a_t)
                # info.append(step_info)
                done = terminated or truncated
                self.model(s_t, a_t, r_tp1, s_tp1)

                q_sa = self.q(s_t, a_t)
                self.Q[s_t][a_t] += self.alpha * (r_tp1 + self.gamma * self.q_max(s_tp1, self.Q) - q_sa)
                s_t = s_tp1
                if done: break




