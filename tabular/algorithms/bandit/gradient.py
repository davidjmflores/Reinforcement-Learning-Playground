import numpy as np

class GradientBanditAgent:
    def __init__(self, k, alpha, use_bl):
        self.k = int(k)
        self.alpha = float(alpha)
        if not (0.0 < self.alpha): # Convergence parameter
            raise ValueError("alpha must be greater than zero.")
        
        self.H = np.zeros(self.k, dtype=float) # Preference init for all actions
        self.pi = np.full(self.k, 1.0 / self.k) # Probability of choosing action a based on preference
        self.r_bar = 0.0 # Average reward for preference calculation. Using incremental update
        self.t = 0 # Timestep counter. For calculating avg reward
        self.use_bl = bool(use_bl)
    
    def soft_max(self):
        h = self.H - np.max(self.H) # SHifts values to all be less than zero except for max(s)
        exp_h = np.exp(h) # raises all to e for getting probability selections
        self.pi = exp_h / np.sum(exp_h) # Probability calculation for all actions in pi

    def select_action(self, rng):
        self.soft_max()
        return int(rng.choice(self.k, p=self.pi)) # Chooses action based on preference 
    
    def update(self, action, reward):
        a = int(action)
        r = float(reward)

        baseline = self.r_bar if self.use_bl else 0.0
        adv = r - baseline

        self.H -= self.alpha * adv * self.pi
        self.H[a] += self.alpha * adv * (1.0 - self.pi[a])

        if self.use_bl:
            self.t += 1
            self.r_bar += (r - self.r_bar) / self.t