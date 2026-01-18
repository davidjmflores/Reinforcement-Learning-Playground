import numpy as np # For argmax, and default_rng

class StationaryKArmedBandit:
    def __init__(self, k, mean_q, sigma_q, sigma_r, seed=None, init_q=None):
        self.k = int(k) # K-number of bandits(actions)
        self.mean_q = mean_q #  Mean for sampling the true action values q*
        self.sigma_q = sigma_q # Standard deviation of q*
        self.sigma_r = sigma_r # Standard deviation of reward
        self.rng = np.random.default_rng(seed) # Random seeding for reproducibility
        self.init_q = init_q
        self.q_star = None # Set of k-normal dist bandits

    def reset(self): # Reset corrosponds to new run or new bandit problem. new set of q*(a)
        if self.init_q is None:
            self.q_star = self.rng.normal(self.mean_q, self.sigma_q, size=self.k) # Init armS W/ sampled values from normal of mean 0 and std 1
        else:
            if np.isscalar(self.init_q):
                self.q_star = np.full(self.k, float(self.init_q), dtype=float) # Initialize all q_star values to same
            else:
                arr = np.asarray(self.init_q, dtype=float)
                if arr.shape != (self.k,):
                    raise ValueError(f"init_q must be scalar or shape ({self.k},), got {arr.shape}")
                self.q_star = arr.copy()

    def step(self, action):
        if self.q_star is None:
            raise RuntimeError("Call reset() before step().") # Prevents calling step before reset
        
        a = int(action) # Selected action at step
        if not (0 <= a < self.k): # Prevent actions out-of-set
            raise ValueError(f"action must be in [0, {self.k-1}], got {a}")
        
        optimal = int(np.argmax(self.q_star)) # For info/debugging
        reward = float(self.rng.normal(self.q_star[a], self.sigma_r)) # Samples reward from N(q*(a), 1)
        
        info = {"optimal_action": optimal, "is_optimal": (a == optimal)}
        return reward, info
    