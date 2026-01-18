import numpy as np # For np.max, zeros, flatnonzeros

class GreedyAgent: # Agent that uses greedy algorithm
    def __init__(self, k, q_init=None):
        self.k = int(k) # Init number of bandits(for action info)
        if q_init is None: self.Q = np.zeros(self.k, dtype=float) # Init all action values to zero
        else: self.Q = np.full(self.k, float(q_init), dtype=float) # Added for potential optimistic initial value testing
        self.N = np.zeros(self.k, dtype=int) # Init selection-counter for each action to zero

    def select_action(self, rng):
        # argmax returns first maximum, not a set of maximums, so need new method bellow
        max_q = np.max(self.Q) # Find max value among actions
        candidates = np.flatnonzero(self.Q == max_q) # collect all actions with values equal to max_q
        return int(rng.choice(candidates)) # randomly select action among candidates
    
    def update(self, action, reward):
        a = int(action) # prevent potential float actions
        self.N[a] += 1 # When action selected, increment selection-counter for it
        step_size = 1.0 / self.N[a] # Increment step size 
        self.Q[a] += step_size * (reward - self.Q[a]) # Using incremental value update on page 31 of Barto and Sutton

