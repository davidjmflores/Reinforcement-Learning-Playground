import numpy as np # For argmax, and default_rng

class NonstationaryKArmedBandit:
    def __init__(self, k, init_val, sigma_r, sigma_n, seed=None):
        self.k = int(k) # K-number of bandits(actions)
        self.init_val = float(init_val) # Initial value for all actions
        self.sigma_n = float(sigma_n) # nonstationary random walk to k-arm bandit values
        self.sigma_r = float(sigma_r) # Standard deviation of reward
        self.rng = np.random.default_rng(seed) # Random seeding for reproducibility
        self.q_star = None # Set of k-normal dist bandits

    def reset(self): # Reset corrosponds to new run or new bandit problem. new set of q*(a)
        self.q_star = np.full(self.k, float(self.init_val), dtype=float) # Initialize all q_star values to same

    def step(self, action):
        if self.q_star is None:
            raise RuntimeError("Call reset() before step().") # Prevents calling step before reset
        
        a = int(action) # Selected action at step
        if not (0 <= a < self.k): # Prevent actions out-of-set
            raise ValueError(f"action must be in [0, {self.k-1}], got {a}")
        
        optimal = int(np.argmax(self.q_star)) # For info/debugging
        reward = float(self.rng.normal(self.q_star[a], self.sigma_r)) # Samples reward from N(q*(a), 1)
                
        self.q_star += self.rng.normal(0.0, self.sigma_n, size=self.k)
        
        info = {"optimal_action": optimal, "is_optimal": (a == optimal)}
        return reward, info