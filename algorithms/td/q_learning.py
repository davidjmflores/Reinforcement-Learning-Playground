# Q-learning (off-policy TD control) for estimating pi
# based on pseudocode on pg. 131

class BehaviorPolicy: # epsilon-greedy
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon

    def sample(self, rng, s, Q):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions available in state {s}")

        if rng.random() < self.epsilon: return actions[rng.integers(len(actions))]

        q_vals = [Q.get(s, {}).get(a, 0.0) for a in actions]
        max_q = max(q_vals)
        greedy_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
        return greedy_actions[rng.integers(len(greedy_actions))]

class QLearning:
    def __init__(self, rng, env, alpha, epsilon, gamma, policy_b=None):
        self.env = env
        self.rng = rng

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        if not 0 < self.alpha <= 1.0: raise ValueError(f"Invalid learning parameter! alpha = {self.alpha}")
        if not 0 <= self.epsilon <= 1.0: raise ValueError(f"Invalid learning parameter! epsilon = {self.epsilon}")
        if not 0 <= self.gamma <= 1.0: raise ValueError(f"Invalid learning parameter! gamma = {self.gamma}")
        
        self.Q = {}

        if policy_b is None: self.policy_b = BehaviorPolicy(self.env, self.epsilon)
        else: self.policy_b = policy_b

    def q(self, s, a):
        if s not in self.Q: self.Q[s] = {}
        if a not in self.Q[s]: self.Q[s][a] = 0.0 # zero init all actions
        return self.Q[s][a]
    
    def run(self, episodes):
        info = []
        for ep in range(episodes):
            s_t, reset_info = self.env.reset()
            info.append(reset_info)
        
            while True:
                a_t = self.policy_b.sample(self.rng, s_t, self.Q)
                s_prime, r, terminated, truncated, step_info = self.env.step(a_t)
                info.append(step_info)
                done = terminated or truncated
                
                q_sa = self.q(s_t, a_t) # used to add s_t and a_t to Q set
                if done: target = r
                else:
                    actions_prime = list(self.env.actions(s_prime))
                    max_q = max(self.Q.get(s_prime, {}).get(a, 0.0) for a in actions_prime)
                    target = r + self.gamma * max_q

                self.Q[s_t][a_t] += self.alpha * (target - q_sa)

                if done: break
                s_t = s_prime
        
        return self.Q, info
            