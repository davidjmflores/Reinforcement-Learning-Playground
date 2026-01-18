import numpy as np # For np.max, zeros, flatnonzeros

class AlphaGreedyAgent: # Agent that uses greedy algorithm
    def __init__(self, k, alpha, q_init=None):
        self.k = int(k) # Init number of bandits(for action info)
        if q_init is None: self.Q = np.zeros(self.k, dtype=float) # Init all action values to zero
        else: self.Q = np.full(self.k, float(q_init), dtype=float) # Added for potential optimistic initial value testing
        self.alpha = alpha
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

    def select_action(self, rng):
        # argmax returns first maximum, not a set of maximums, so need new method bellow
        max_q = np.max(self.Q) # Find max value among actions
        candidates = np.flatnonzero(self.Q == max_q) # collect all actions with values equal to max_q
        return int(rng.choice(candidates)) # randomly select action among candidates
    
    def update(self, action, reward):
        a = int(action)
        self.Q[a] += self.alpha * (reward - self.Q[a]) # Using incremental value update on page 31 of Barto and Sutton

