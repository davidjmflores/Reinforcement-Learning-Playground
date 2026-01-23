class TabularStochasticPolicy:
    def __init__(self, states, actions):
        self.actions = list(actions)
        p = 1.0 / len(self.actions)
        self.pi_table = {s: {a: p for a in self.actions} for s in states}

    def pi(self, s):
        return self.pi_table[s]

    def snapshot(self):
        return {s: probs.copy() for s, probs in self.pi_table.items()}

    def set_greedy(self, s, greedy_actions):
        if not greedy_actions:
            raise ValueError(f"set_greedy called with empty greedy_actions for state {s}")

        for a in self.actions:
            self.pi_table[s][a] = 0.0

        p = 1.0 / len(greedy_actions)
        for a in greedy_actions:
            self.pi_table[s][a] = p
