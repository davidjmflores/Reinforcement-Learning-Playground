import numpy as np

class FirstVisitMCPrediction:
    def __init__(self, env, policy, v0, gamma):
        self.policy = policy
        self.env = env
        self.v = {s: v0 for s in self.env.states()}
        self.returns = np.zeros(self.env.states(), dtype=int)
        self.gamma = gamma

    def policy_sample(self, s, rng):
        return int(rng.choice(self.env.actions(s), p=self.policy.pi(s)))

    def episode(self, rng):
        s = self.env.s_init
        episode = []
        while s not in self.env.terminal_state: # not defined. Not sure if this is the best way
            a = self.policy_sample(s, rng)
            s_prime, r = self.env.step(s, a)
            episode.append((s, a, r))
            s = s_prime
        return episode

    def prediction(self, episodes):
        '''
        Docstring for prediction

        Loop forever(for each episode):
            Generate an episode following pi: S0, A0, R1, S1, A1, R2, ..., ST-1, AT-1, RT
            G <- 0
            Loop for each step of episode, t=T-1,T-2,...,0:
            G <- G * gamma + Rt+1
            Unless St appears in S0, S1, ..., St-1
                Append G to Returns(St)
                V(St) <- Average(Returns(St))
        
        :param self: Pg. 92 pseudo code
        '''

        i = 0
        while i < episodes:
            episode = self.episode()
            for s in self.episode()[0]: # for all states in that episode
                G = 0
                n = 0
                for step in reversed(episode): # where each step is a (S, A, R)
                    n += 1
                    if step[0] == s:
                        self.returns[s] = G
                        self.v[s] = G / n
                    else: G = G * self.gamma + step[2]
            i+=1
        return self.returns, self.v


