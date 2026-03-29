class DynaMaze:
    def __init__(self):

        self.rows, self.cols = 6, 9
        self.null_states = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]
        self.start = (2, 0)
        self.goal = (0, 8)

        self._actions = [
            (-1, 0), # up
            ( 1, 0), # down
            ( 0,-1), # left
            ( 0, 1) # right
        ]

        self._states = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.null_states
        ]

        self.curr_state = self.start
    
    def actions(self, s=None): return self._actions
    
    def states(self): return self._states

    def reset(self, seed=None):
        self.curr_state = self.start
        return self.curr_state, {}

    def step(self, a):
        r = 0
        row, col = self.curr_state
        dr, dc = a

        r_prime = min(self.rows - 1, max(0, row + dr))
        c_prime = min(self.cols - 1, max(0, col + dc))
        
        s_prime = (r_prime, c_prime)

        if s_prime == self.goal: r = 1
        if not s_prime in self.null_states: self.curr_state = s_prime
        terminated = (s_prime == self.goal)

        return self.curr_state, r, terminated, False, {}


