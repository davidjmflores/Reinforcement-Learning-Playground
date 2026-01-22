class GridworldTerminating:
    def __init__(self):
        # Below values are as described in Example 4.1 of Chapter 4 in Barto & Sutton's RL textbook.
        self.n_rows = 4
        self.n_cols = 4
        self.terminal_state_1 = (3,3)
        self.terminal_state_2 = (0,0)

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

        if s in (self.terminal_state_1, self.terminal_state_2): return s, 0

        s_prime = (s[0] + a[0], s[1] + a[1])
        # Check if action leads agent out-of-bounds.
        if not (0 <= s_prime[0] < self.n_rows and 0 <= s_prime[1] < self.n_cols):
            return s, -1
        
        return s_prime, -1
    # For state enumeration in algorithms like value iteration.
    # So that we don't have to rederive states in each algorithm
    # or store states in each algorithm.
    def states(self):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                yield (r, c)
    # Implemented for iterative policy evaluation. Though, useful for most algs in ch.4 and ch. 5
    def transitions(self, s, a):
        s_prime, r = self.step(s, a)
        return [(1.0, s_prime, r)]