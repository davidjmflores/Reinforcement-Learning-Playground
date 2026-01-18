import numpy as np # For np.max, zeros, flatnonzeros

class EpsilonGreedyAgent: # Agent that uses an epsilon greedy algorithm
    def __init__(self, k, epsilon, q_init=None):
        self.k = int(k) # Init number of bandits(for action info)
        self.epsilon = float(epsilon) # Probability of exploratory action selection
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0,1]")
        if q_init is None: self.Q = np.zeros(self.k, dtype=float) # make all initialn action values zero
        else: self.Q = np.full(self.k, float(q_init), dtype=float) # Added for potential optimistic initial value testing
        self.N = np.zeros(self.k, dtype=int) # Initialize selection-counter to zero for each action(Used for updating value)

    def select_action(self, rng):
        # argmax returns first maximum, not a set of maximums, so need new method bellow
        if rng.random() < self.epsilon:
            return int(rng.integers(self.k))

        max_q = np.max(self.Q) # Find max value among actions
        candidates = np.flatnonzero(self.Q == max_q) # collect all actions with values equal to max_q
        return int(rng.choice(candidates)) # randomly select action among candidates
    
    def update(self, action, reward):
        a = int(action)
        self.N[a] += 1 # When action selected, increment selection-counter for it
        step_size = 1.0 / self.N[a] # Increment step size 
        self.Q[a] += step_size * (reward - self.Q[a]) # Using incremental value update on page 31 of Barto and Sutton

