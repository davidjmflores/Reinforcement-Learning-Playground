import numpy as np # For np.max, zeros, flatnonzeros

class AlphaEpsilonGreedyAgent: # Agent that uses an epsilon greedy algorithm
    def __init__(self, k, epsilon, alpha):
        self.k = int(k) # Init number of bandits(for action info)
        self.epsilon = float(epsilon) # Probability of exploratory action selection
        self.alpha = float(alpha) # constant step-size parameter
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0,1]")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.Q = np.zeros(self.k, dtype=float) # make all initialn action values zero
    def select_action(self, rng):
        # argmax returns first maximum, not a set of maximums, so need new method bellow
        if rng.random() < self.epsilon:
            return int(rng.integers(self.k))

        max_q = np.max(self.Q) # Find max value among actions
        candidates = np.flatnonzero(self.Q == max_q) # collect all actions with values equal to max_q
        return int(rng.choice(candidates)) # randomly select action among candidates
    
    def update(self, action, reward):
        a = int(action)
        self.Q[a] += self.alpha * (reward - self.Q[a])
