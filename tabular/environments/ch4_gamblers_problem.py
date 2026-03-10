# Gambler can bet coins on outcomes of a sequence of coin flips.
# If heads, he wins as many dollars as he's staked on that flip
# if tails, loses stake.
# Game ends when gambler wins by reaching goal of $100 or loses by running out of money

# undiscounted episodic and finite MDP.

# r = 0 on all transitions except those that lead to goal +1
# v(s) = p(winning | s)
# pi(s) = mapping from levels of capital to stakes
# pi*(s) maximizing p(reaching goal)

class GamblersProblem:
    def __init__(self):
        # Problem constraints
        self.winning_val = 100 # Value gambler needs to earn in order to win
        self.p_h = 0.4 # Prob coin comes up heads

        self._states = (range(0, self.winning_val + 1)) # Gambler's capital
        self._actions = {}
        for s in self._states:
            self._actions[s] = tuple(range(1, min(s, self.winning_val - s) + 1)) # Gambler's stakes
        
        self.transitions_cache = {} # cache for transitions(s, a)

    
    def states(self):
        return self._states
    
    def actions(self, s):
        return self._actions[s]
    
    def transitions(self, s, a):
        # two possibilities 
        if a not in self._actions[s]:
            raise ValueError(f"Invalid action {a}. Must be one of {self._actions[s]}")
        
        key = (s, a)
        cached = self.transitions_cache.get(key)
        if cached is not None:
            return cached # already tuple of (p, s_prime, r_bar)
        
        transitions = []

        # Outcome tails - lose
        s_prime = s - a
        if s_prime == 0: transitions.append((1 - self.p_h, 0, 0))
        else: transitions.append((1 - self.p_h, s_prime, 0))

        # Outcome heads - wins
        s_prime = s + a
        if s_prime >= self.winning_val:
            transitions.append((self.p_h, self.winning_val, 1))
        else: transitions.append((self.p_h, s_prime, 0))

        self.transitions_cache[key] = transitions  
        return transitions


