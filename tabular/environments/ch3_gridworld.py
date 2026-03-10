class Gridworld:
    def __init__(self):
        # Below values are as described in Example 3.5 of Chapter 3 in Barto & Sutton's RL textbook.
        self.n_rows = 5
        self.n_cols = 5
        self.A = (0, 1)
        self.A_prime = (4,1)
        self.r_A_prime = 10
        self.B = (0, 3)
        self.B_prime = (2, 3)
        self.r_B_prime = 5
        self.r_off_grid = -1

        self.actions = [
            (-1, 0), # Up
            ( 1, 0), # Down
            ( 0,-1), # Left
            ( 0, 1)  # Right
        ]
    # (s, a) -> (s', r)
    def step(self, s, a):
        if a not in self.actions:
            raise ValueError(f"Invalid action {a}. Must be one of {self.actions}")
        s = tuple(s)

        if s == self.A: return self.A_prime, self.r_A_prime
        if s == self.B: return self.B_prime, self.r_B_prime

        s_prime = (s[0] + a[0], s[1] + a[1])
        # Check if action leads agent out-of-bounds.
        if not (0 <= s_prime[0] < self.n_rows and 0 <= s_prime[1] < self.n_cols):
            return s, self.r_off_grid
        
        return s_prime, 0
    # For state enumeration in algorithms like value iteration.
    # So that we don't have to rederive states in each algorithm
    # or tore states in each algorithm.
    def states(self):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                yield (r, c)
    
    # Implemented for value updates. specifically, providing to the algorithm p(s',r|s,a) for all s' and r
    def transitions(self, s, a):
        s_prime, r = self.step(s, a)
        return [(1.0, s_prime, r)]

        