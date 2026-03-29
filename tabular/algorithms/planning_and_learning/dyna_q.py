class EpsilonGreedy:
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon

    def sample(self, rng, s, Q):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions for state: {s}")

        if rng.random() < self.epsilon: return actions[rng.integers(len(actions))]

        q_vals = [Q.get(s, {}).get(a, 0.0) for a in actions]
        q_max = max(q_vals)
        greedy_actions = [a for a, q in zip(actions, q_vals) if q == q_max]
        return greedy_actions[rng.integers(len(greedy_actions))]

class DynaQ:
    def __init__(self, rng, env, epsilon, gamma, alpha, n):
        self.rng = rng
        self.env = env
        self.gamma = gamma 
        self.alpha = alpha
        self.n = n # represents iterations of planning per step (not t-steps)

        self.Q = {}
        self.model = {} # model[s][a] = (r, s_prime, done)

        self.behavior = EpsilonGreedy(self.env, epsilon)
    
    def q_max(self, s):
        actions = list(self.env.actions(s))
        if not actions: return 0.0

        return max(self.Q.get(s, {}).get(a, 0.0) for a in actions)
    
    def update_q(self, s, a, r, s_prime, done):
        target = r if done else r + self.gamma * self.q_max(s_prime)
        self.Q.setdefault(s, {})
        self.Q[s].setdefault(a, 0.0)
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])
    
    def update_model(self, s, a, r, s_prime, done):
        self.model.setdefault(s, {})
        self.model[s][a] = (r, s_prime, done)

    def planning_step(self):
        if not self.model:
            return

        for _ in range(self.n):
            states = list(self.model.keys())
            s = states[self.rng.integers(len(states))]

            actions_taken_in_s = list(self.model[s].keys())
            a = actions_taken_in_s[self.rng.integers(len(actions_taken_in_s))]

            r, s_prime, done = self.model[s][a]
            self.update_q(s, a, r, s_prime, done)

    def run(self, episodes):
        # info = []
        for _ in range(episodes):
            s_t, reset_info = self.env.reset(self.rng)
            done = False
            # info.append(reset_info)

            while not done:
                a_t = self.behavior.sample(self.rng, s_t, self.Q)

                s_tp1, r_tp1, terminated, truncated, step_info = self.env.step(s_t, a_t)
                # info.append(step_info)
                done = terminated or truncated

                self.update_q(s_t, a_t, r_tp1, s_tp1, done)
                self.update_model(s_t, a_t, r_tp1, s_tp1, done)
                self.planning_step()

                s_t = s_tp1

        return self.Q