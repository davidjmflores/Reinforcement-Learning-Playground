class BlockingMaze:
    def __init__(self):
        self.rows, self.cols = 6, 9
        self.start = (5, 3)
        self.goal = (0, 8)

        self._actions = [
            (-1, 0), # up
            ( 1, 0), # down
            ( 0,-1), # left
            ( 0, 1)  # right
        ]

        self.initial_walls = {(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)}
        self.blocked_walls = {(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)}

        self.curr_state = self.start
        self.total_steps = 0

    def states(self):
        walls = self.current_walls()
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in walls
        ]

    def actions(self, s=None): return self._actions

    def current_walls(self): return self.initial_walls if self.total_steps < 1000 else self.blocked_walls

    def reset(self, seed=None): 
        self.curr_state = self.start
        return self.curr_state, {}

    def step(self, a):
        row, col = self.curr_state
        dr, dc = a

        r_prime = min(self.rows - 1, max(0, row + dr))
        c_prime = min(self.cols - 1, max(0, col + dc))
        s_prime = (r_prime, c_prime)

        if s_prime not in self.current_walls(): self.curr_state = s_prime

        reward = 1 if self.curr_state == self.goal else 0
        terminated = (self.curr_state == self.goal)

        self.total_steps += 1
        return self.curr_state, reward, terminated, False, {}

    

