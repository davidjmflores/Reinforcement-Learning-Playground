# Normal gridworld 7x10 except that there is a crosswind running upwards
# through the middle of the grid
# columns:        [4, 5, 6, 7, 8, 9] 
# wind intensity: [1, 1, 1, 2, 2, 1]
# actions: [up, down, left, right]
# In the wind-effect states, next states are shifted upward by the wind
# Undiscounted episodic task, w/ constant r = -1 until goal reached.

class WindyGridworld:
    def __init__(self):
        self._rows, self._cols = 7, 10
        self._actions = [(-1, 0), ( 1, 0), ( 0,-1), ( 0, 1)]
        self._wind_strength_by_col = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self._start = (3, 0) # start position
        self._goal  = (3, 7)
        self._curr_state = self._start # internalized for compatibility w/ gym

        self._states = []
        for r in range(self._rows):
            for c in range(self._cols):
                self._states.append((r, c))
    
    def actions(self, s=None): return self._actions

    # Added for MC
    def states(self): return [(r, c) for r in range(self._rows) for c in range(self._cols)]

    def reset(self): 
        self._curr_state = self._start
        return self._start, {} # Gym-like returns (obs0, reset_info)

    def step(self, a): # state not part of step as compliance to gym
        r, c = self._curr_state
        dr, dc = a
        wind = self._wind_strength_by_col[c]
        r_prime = min(self._rows - 1, max(0, r + dr - wind))
        c_prime = min(self._cols - 1, max(0, c + dc))
        
        self._curr_state = (r_prime, c_prime)
        terminated = (self._curr_state == self._goal)
        
        
        return self._curr_state, -1, terminated, False, {} # Gym-like returns (obs, r, done, truncate, step_info)
