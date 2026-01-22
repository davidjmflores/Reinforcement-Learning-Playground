class UniformRandom:
    def __init__(self, actions):
        self.actions = list(actions)
        prob = 1.0 / len(self.actions)
        self.action_probs = {a: prob for a in self.actions}

    # s included for future environments where different actions are available in different states(where A(s) needed)
    def pi(self, s):
        return self.action_probs
