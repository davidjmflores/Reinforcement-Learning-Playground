# Sarsa (On-policy TD control) for estimating Q
class EpsilonGreedyPolicy: # Behavioral Policy
    def __init__(self, env, epsilon, rng):
       self.env = env
       self.epsilon = epsilon
       self.rng = rng
       self.pi_table = {}

       for s in env.states():
          actions = list(self.env.actions(s))
          if not actions:
             self.pi_table[s] = {}
             continue
          
          # set all A to have p = epsilon / |A(s)|
          p = self.epsilon / len(actions)
          self.pi_table[s] = {a: p for a in actions}

          # randomly select an init greedy action
          a_greedy = rng.choice(actions)
          self.pi_table[s][a_greedy] += 1.0 - self.epsilon
    
    def pi(self, s): return self.pi_table[s]
    
    def make_greedy(self, s, greedy_action):
       actions = list(self.env.actions(s))
       if not actions: return

       base = self.epsilon / len(actions)
       dist = {a: base for a in actions}
       dist[greedy_action] += 1.0 - self.epsilon
       self.pi_table[s] = dist

    def policy_sample(self, s):
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions available in state {s}")

        dist = self.pi(s)
        probs = [dist.get(a, 0.0) for a in actions]
        total = sum(probs)
        if total <= 0: probs = [1.0 / len(actions)] * len(actions)
        else: probs = [x / total for x in probs]
        return self.rng.choice(actions, p=probs)

class Sarsa:
    def __init__(self, env, gamma, alpha, epsilon, rng, threshold):
        self.env = env
        self.epsilon = epsilon
        self.policy = EpsilonGreedyPolicy(env, epsilon, rng)
        self.gamma = gamma
        self.alpha = alpha
        if not 0 < self.alpha <= 1: raise ValueError(f"Alpha value not within bounds (0, 1]: alpha = {self.alpha}")
        self.threshold = threshold
        
        # I feel like  should stop using Q initialization like this. When states become massive its
        # impractical to initialize for ll states. I should just initialize when reached, right?
        self.Q = {s: {a: 0.0 for a in self.env.actions(s)} for s in self.env.states()}


    def run(self, episodes):
        for _ in range(episodes):
           info = {}
           s_t = self.env.reset()
           a_t = self.policy.sample_policy(s_t)
           
           while True:
            s_prime, r, done, i = self.env.step(a_t)
            a_prime = self.policy.sample_policy(s_prime)
            
            # This can be done instead of Q init in init, but idk if its computationally better
            if self.Q[s_t][a_t] is None:
               self.Q[s_t][a_t] = 0
            if self.Q[s_prime][a_prime] is None:
               self.Q[s_prime][a_prime] = 0

            self.Q[s_t][a_t] += self.alpha * (r + self.gamma * self.Q[s_prime][a_prime] - self.Q[s_t][a_t])

            if done: break

            s_t = s_prime
            a_t = a_prime
            info.append(i)
        
        return self.Q, info