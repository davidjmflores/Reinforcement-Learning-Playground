# Policy used for policy improvement in PI, VI, and MCES

class TabularStochasticPolicy:
    def __init__(self, env):
        self.env = env
        self.pi_table = {}

        for s in env.states():
            actions = list(env.actions(s))
            if not actions:
                self.pi_table[s] = {}
                continue

            p = 1.0 / len(actions)
            self.pi_table[s] = {a: p for a in actions}

    def pi(self, s):
        return self.pi_table[s]

    def set_greedy(self, s, greedy_actions):
        if not greedy_actions:
            raise ValueError(f"set_greedy called with empty greedy_actions for state {s}")

        actions = list(self.env.actions(s))
        if not actions:
            self.pi_table[s] = {}
            return

        # zero only legal actions
        self.pi_table[s] = {a: 0.0 for a in actions}

        p = 1.0 / len(greedy_actions)
        for a in greedy_actions:
            if a not in self.pi_table[s]:
                raise ValueError(f"Greedy action {a} not legal in state {s}")
            self.pi_table[s][a] = p

class JacksTabularStochasticPolicy:
    def __init__(self, env):
        self.env = env
        self._pi = {}

        for s in env.states():
            self._pi[s] = {0: 1.0}
    
    def pi(self, s):
        return self._pi[s]
    
    def set_greedy(self, s, greedy_actions):
        p = 1.0 / len(greedy_actions)
        self._pi[s] = {a: p for a in greedy_actions}

    def snapshot(self):
        # used by logging in PI
        return {s: dict(dist) for s, dist in self._pi.items()}
    
class GamblersTabularPolicy:
    def __init__(self, env):
        self.env = env
        self._pi = {}

        # Initialize a valid placeholder policy for every state
        for s in env.states():
            actions = env.actions(s)
            self._pi[s] = {actions[0]: 1.0} if actions else {}

    def pi(self, s):
        return self._pi[s]
    
    def set_greedy(self, s, greedy_actions):
        if not greedy_actions:
            self._pi[s] = {}
            return
        p = 1.0 / len(greedy_actions)
        self._pi[s] = {a: p for a in greedy_actions}

    def snapshot(self):
        # used by logging in PI
        return {s: dict(dist) for s, dist in self._pi.items()}

