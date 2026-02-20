# gridworld example 4 x 12 grid. grid (3, 0) is the start and grid (3, 11) is goal
# grids (3, 1-10) are cliff grids. Landing on these grids provides r = -100 and sends
# the agent instantly back to the start. every other action provides r = -1

class CliffWalking:
    def __init__(self):
        # env
        self.rows, self.cols = 4, 12
        self._states = [(r, c) for r in range(self.rows) for c in range(self.cols)]

        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = {(3, c) for c in range(1, 11)}

        self._actions = [(-1, 0), (+1, 0), ( 0,-1), ( 0, +1)] # up down left right
        self.curr_state = self.start
    
    def actions(self, s=None): return self._actions

    def states(self): return self._states # Added for MC

    def reset(self, seed=None): # rng for Gym
        self.curr_state = self.start
        return self.curr_state, {}
    
    def step(self, a):
        r = -1
        terminated = False

        row, col = self.curr_state
        dr, dc = a

        row_prime = min(self.rows - 1, max(row + dr, 0))
        col_prime = min(self.cols - 1, max(col + dc, 0))
        self.curr_state = (row_prime, col_prime)

        if self.curr_state in self.cliff: 
            r = -100
            self.curr_state = self.start
        terminated = (self.curr_state == self.goal)
        
        return self.curr_state, r, terminated, False, {}

