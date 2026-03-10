# Off-policy MC predicition (policy evaluation) for estimating Q roughly equal to q_pi

class TargetPolicy: # epsilon-soft policy from on_policy_fvmc.py
    def __init__(self, env, epsilon, rng):
        self.pi_table = {}

        for s in env.states():
            actions = list(env.actions(s))
            if not actions:
                self.pi_table[s] = {}
                continue
            
            # set all actions to have p = epsilon / |A(s)|
            p = epsilon / len(actions)
            self.pi_table[s] = {a: p for a in actions}

            # Randomly select an initial greedy action
            a_greedy = rng.choice(list(actions))
            self.pi_table[s][a_greedy] += 1.0 - epsilon
    
    def pi(self, s): return self.pi_table[s]
    
    # no policy sample needed as we're only trying to learn the state values given episodes ran following another policy
    
class BehaviorPolicy: # equiprobability policy(arbitrary)
    def __init__(self, env):
        self.env = env
        self.pi_table = {}

        for s in self.env.states():
            actions = list(self.env.actions(s))
            if not actions:
                self.pi_table[s] = {}
                continue

            p = 1 / len(actions)
            self.pi_table[s] = {a: p for a in actions}

    def pi(self, s): return self.pi_table[s]
    
    # previously placed policy_sample() in algorithm but it seemed to be better fit under policy class
    def policy_sample(self, s, rng): # using this policy for episode generation so policy sampling needed.
        actions = list(self.env.actions(s))
        if not actions: raise ValueError(f"No actions available in state {s}")
        
        dist = self.pi(s)
        probs = [dist.get(a, 0.0) for a in actions]
        total = sum(probs)
        if total <= 0: probs = [1.0 / len(actions)] * len(actions)
        else: probs = [p / total for p in probs]
        return rng.choice(actions, p=probs)

# algorithm
class OffPolicyMCP:
    def __init__(self, env, policy_t, policy_b, gamma):
        self.env = env
        self.policy_t = policy_t
        self.policy_b = policy_b
        self.gamma = gamma

        self.Q = {s: {a: 0.0 for a in self.env.actions(s)} for s in self.env.states()}
        self.C = {s: {a: 0.0 for a in self.env.actions(s)} for s in self.env.states()}
    
    def episode(self, rng):
        episode = []

        s0, r0, done = self.env.reset(rng)
        if done:
            episode.append((s0, None, r0))
            return episode
        
        s = s0
        while True:
            a = self.policy_b.policy_sample(s, rng)
            s_prime, r, done = self.env.step(s, a, rng)
            episode.append((s, a, r))

            if done: break
            s = s_prime
        return episode

    def run(self, episodes, rng): # or should I call it run()?
        for _ in range(episodes):
            episode = self.episode(rng)

            T = len(episode)

            G = 0.0
            W = 1.0

            for t in range(T-1, -1, -1):  
                if W == 0.0: break

                s_t = episode[t][0]
                a_t = episode[t][1]
                if a_t is None: continue
                r = episode[t][2]

                pi_prob = self.policy_t.pi(s_t).get(a_t, 0.0)
                if pi_prob == 0.0: break
                b_prob = self.policy_b.pi(s_t).get(a_t, 0.0)
                if b_prob == 0.0: raise ValueError(f"Coverage violated.")
        
                G = self.gamma * G + r
                self.C[s_t][a_t] += W
                self.Q[s_t][a_t] +=  (W / self.C[s_t][a_t]) * (G - self.Q[s_t][a_t]) # incremental update
                W *= pi_prob / b_prob
        
        return self.Q

