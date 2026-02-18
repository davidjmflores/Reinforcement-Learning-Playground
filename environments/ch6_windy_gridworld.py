# Normal gridworld 7x10 except that there is a crosswind running upwards
# through the middle of the grid
# columns:        [4, 5, 6, 7, 8, 9] 
# wind intensity: [1, 1, 1, 2, 2, 1]
# actions: [up, down, left, right]
# In the wind-effect states, next states are shifted upward by the wind
# Undiscounted episodic task, w/ constant r = -1 until goal reached.

class WindyGridworld:
    def __init__(self):
        self._rows = 7
        self._cols = 10

        self._states = []
        for r in range(self._rows):
            for c in range(self._cols):
                self._states.append((r, c))
        
        self._actions = [
            (0, -1), # up
            (0,  1), # down
            (-1, 0), # left
            ( 1, 0)  # right
        ]
        
        # wind dynamics
        _windy_cols =    [3, 4, 5, 6, 7, 8] # columns w/ wind (starting at zero)
        _wind_strength = [1, 1, 1, 2, 2, 1] # said columns wind strength
        self._windy_col = {c: {s: s for s in _wind_strength} for c in _windy_cols}

        self._start = (3, 0) # start position
        self._goal  = (3, 7)
        self._curr_state = self._start # internalized for compatibility w/ gym
        self._info = []
    
    def states(self): return self._states

    def actions(self, s):
        actions = []
        for a in self._actions:
            if -1 < (s[0] + a[0] and s[1] + a[0]) < self._rows:
                actions.append(a)

    def reset(self): 
        self._curr_state = self._start
        self._info = []
        return self._start

    def step(self, a): # state not part of step as compliance to gym
        s_prime = ()
        r = -1
        done = False
        # will need to include logic for windy states 
        if self._curr_state[1] in self._windy_col:
            s_prime[0] = self._curr_state[0] + a[0] + self._windy_col[self._curr_state] # row
            s_prime[1] = self._curr_state[1] + a[1]
        else:
            s_prime[0] = self._curr_state[0] + a[0]
            s_prime[1] = self._curr_state[1] + a[1]
        
        if s_prime == self._goal:
            r = 0
            done = True
        
        return s_prime, r, done, self._info
