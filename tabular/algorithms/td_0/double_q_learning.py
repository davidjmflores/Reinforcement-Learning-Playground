class EpsilonGreedyPolicy: # epsilon-greedy
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon

    def sample(self, rng, s_t, Q1, Q2):
        actions = list(self.env.actions(s_t))
        if not actions: raise ValueError(f"No actions available in state {s_t}")

        if rng.random() < self.epsilon: return actions[rng.integers(len(actions))]

        q_vals = [Q1.get(s_t, {}).get(a, 0.0) + Q2.get(s_t, {}).get(a, 0.0) for a in actions]
        max_q = max(q_vals)
        greedy_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
        return greedy_actions[rng.integers(len(greedy_actions))]


class DoubleQLearning:
    def __init__(self, rng, env, gamma, alpha, epsilon):
        self.env = env
        self.rng = rng
        self.gamma= gamma
        self.alpha = alpha
        if not (0 < self.alpha <= 1.0): raise ValueError(f"Invalid parameter value: alpha = {self.alpha}")
        self.epsilon = epsilon
        if not (0 < self.epsilon <= 1.0): raise ValueError(f"Invalid parameter: epsilon = {self.epsilon}")

        self.policy = EpsilonGreedyPolicy(self.env, self.epsilon)

        self.Q1 = {}
        self.Q2 = {}
    
    def q(self, Q, s, a):
        Q.setdefault(s, {})
        Q[s].setdefault(a, 0.0)
        return Q[s][a]
    
    def argmax_action(self, Q, s, actions):
        vals = [Q.get(s, {}).get(a, 0.0) for a in actions]
        m = max(vals)
        greedy = [a for a, v in zip(actions, vals) if v == m]
        return greedy[self.rng.integers(len(greedy))]
    
    def run(self, episodes):
        for ep in range(episodes):
            s_t, reset_info = self.env.reset()
            # info.append(reset_info)

            while True:
                a_t = self.policy.sample(self.rng, s_t, self.Q1, self.Q2)
                s_prime, r, terminated, truncated, step_info = self.env.step(a_t)
                done = terminated or truncated
                # info.append(step_info)

                if self.rng.random() < 0.5: Q_upd, Q_eval = self.Q1, self.Q2
                else: Q_upd, Q_eval = self.Q2, self.Q1

                q_sa = self.q(Q_upd, s_t, a_t)

                if done: target = r
                else:
                    actions_prime = list(self.env.actions(s_prime))
                    a_star = self.argmax_action(Q_upd, s_prime, actions_prime)
                    target = r + self.gamma * Q_eval.get(s_prime, {}).get(a_star, 0.0)

                Q_upd[s_t][a_t] += self.alpha * (target - q_sa)

                if done: break
                s_t = s_prime
        
        return self.Q1, self.Q2
