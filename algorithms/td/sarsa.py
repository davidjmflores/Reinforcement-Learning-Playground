# Sarsa (On-policy TD control) for estimating Q
class EpsilonGreedyPolicy: # Behavioral Policy
   def __init__(self, env, epsilon):
      self.env = env
      self.epsilon = epsilon

   def sample(self, rng, s, Q):
      actions = list(self.env.actions(s))
      if not actions: raise ValueError(f"No actions available in state {s}")

      if rng.random() < self.epsilon: return actions[rng.integers(len(actions))]
        
      # exploit
      q_vals = [Q.get(s, {}).get(a, 0.0) for a in actions]
      max_q = max(q_vals)
      greedy_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
      return greedy_actions[rng.integers(len(greedy_actions))]

class Sarsa:
   def __init__(self, env, gamma, alpha, epsilon, rng):
      self.env = env
      self.epsilon = epsilon
      self.policy = EpsilonGreedyPolicy(env, epsilon)
      self.gamma = gamma
      self.alpha = alpha
      self.rng = rng
      self.total_steps = 0
      if not 0 < self.alpha <= 1: raise ValueError(f"Alpha value not within bounds (0, 1]: alpha = {self.alpha}")
      self.Q = {}

   def q(self, s, a):
      if s not in self.Q: self.Q[s] = {}
      if a not in self.Q[s]: self.Q[s][a] = 0.0
      return self.Q[s][a]

   def run(self, episodes):
      episode_end_steps = []
      for ep in range(episodes):
         s_t, _ = self.env.reset()
         a_t = self.policy.sample(self.rng, s_t, self.Q)

         while True:
            s_prime, r, terminated, truncated, _ = self.env.step(a_t)
            done = terminated or truncated

            q_sa = self.q(s_t, a_t)
            if done: target = r
            else:
               a_prime = self.policy.sample(self.rng, s_prime, self.Q)
               target = r + self.gamma * self.q(s_prime, a_prime)

            self.Q[s_t][a_t] = q_sa + self.alpha * (target - q_sa)

            self.total_steps += 1

            if done: 
               episode_end_steps.append(self.total_steps)
               break
            
            s_t, a_t = s_prime, a_prime
      
      return self.Q, episode_end_steps