import numpy as np # For np.max, zeros, flatnonzeros

class UpperConfidenceBound: # Agent that selects the non-greedy actions according to their potential for actually being optimal
    def __init__(self, k, c, q_init=None):
        self.k = int(k) # Init number of bandits(for action info)
        self.c = float(c) # Probability of exploratory action selection
        if self.c < 0.0:
            raise ValueError("c must be >= 0")
        if q_init is None: self.Q = np.zeros(self.k, dtype=float) # make all initialn action values zero
        else: self.Q = np.full(self.k, float(q_init), dtype=float) # Added for potential optimistic initial value testing
        self.N = np.zeros(self.k, dtype=int) # Initialize selection-counter to zero for each action(Used for updating value)
        self.t = 0 # for total action selection

    def select_action(self, rng):
        self.t += 1 # increment time step

        # If any action hasn't been tried, try one of them first
        untried = np.flatnonzero(self.N == 0)
        if untried.size > 0:
            return int(rng.choice(untried))

        # UCB values
        bonus = self.c * np.sqrt(np.log(self.t) / self.N)
        ucb_values = self.Q + bonus

        # random tie-break argmax
        max_val = np.max(ucb_values)
        candidates = np.flatnonzero(ucb_values == max_val)
        return int(rng.choice(candidates))
    
    def update(self, action, reward):
        a = int(action)
        self.N[a] += 1 # When action selected, increment selection-counter for it
        step_size = 1.0 / self.N[a] # Increment step size 
        self.Q[a] += step_size * (reward - self.Q[a]) # Using incremental value update on page 31 of Barto and Sutton

