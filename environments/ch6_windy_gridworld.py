import numpy as np
class WindyGridworld:
    def __init__(self):
        self._rows, self._cols = 7, 10
        self._actions = [(-1, 0), ( 1, 0), ( 0,-1), ( 0, 1), # up, down, left, right
                         (-1,-1), (-1, 1), ( 1,-1), ( 1, 1), # top-left, top-right, bottom-left, bottom-right
                         ( 0, 0)] # stay still
        
        self._wind_strength_by_col = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self._start = (3, 0) # start position
        self._goal  = (3, 7)
        self._curr_state = self._start # internalized for compatibility w/ gym

        self._states = [(r, c) for r in range(self._rows) for c in range(self._cols)]

        self.np_random = np.random.default_rng()

    def actions(self, s=None): return self._actions

    def states(self): return self._states # Added for MC

    def reset(self, seed=None): 
        if seed is not None: self.np_random = np.random.default_rng(seed)
        self._curr_state = self._start
        return self._start, {} # Gym-like returns (obs0, reset_info)

    def wind_effect(self, col):
        w = self._wind_strength_by_col[col]
        if w == 0: return 0
        dev = self.np_random.choice([-1, 0, 1])
        return w + dev
    
    def step(self, a): # state not part of step as compliance to gym
        r, c = self._curr_state
        dr, dc = a
        wind = self.wind_effect(c)
        r_prime = min(self._rows - 1, max(0, r + dr - wind))
        c_prime = min(self._cols - 1, max(0, c + dc))
        
        self._curr_state = (r_prime, c_prime)
        terminated = (self._curr_state == self._goal)
        
        
        return self._curr_state, -1, terminated, False, {} # Gym-like returns (obs, r, done, truncate, step_info)
