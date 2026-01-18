# 5x5 gridworld 
# cell at 1-2 is called A and leads agent to A'(5-2) giving reward +10
# cell at 1-4 is called B and leads agent to B'(3-4) giving reward +5 
# The agent can take actions up, down, left, and right.
# Actions that would take agent off the grid, willl leave the agents pos unchanged but inflict reward of -1
# Other actions result in reward of zero
# will probably need transition matrices and states represented by trnsition matrices

class Gridworld():
    def __init__(self, n_rows, n_cols):
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.A = {0, 1}
        self.A_prime = {4,1}
        self.r_A_prime = 10
        self.B = {0, 3}
        self.B_prime = {2, 3}
        self.r_B_prime = 5
        self.r_off_grid = -1
        self.r_other = 0

    def step(self, state, action):
        s = tuple(state)
        a = tuple(action)
        s_prime = s - a
        r = int(0)
        if s_prime[0] == -1 or s_prime[0] == self.n_rows: s_prime = s, r = self.r_off_grid; return s_prime, r
        elif s_prime[1] == -1 or s_prime[1] == self.n_cols: s_prime = s, r = self.r_off_grid; return s_prime, r
        elif s == self.A: s_prime = self.A_prime, r = self.r_A_prime; return s_prime, r
        elif s == self.B: s_prime = self.B_prime, r = self.r_B_prime; return s_prime, r
        else: r = self.r_other; return s_prime, r