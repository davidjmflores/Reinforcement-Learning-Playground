import math

class JacksCarRental:
    def __init__(self):
        self.max_capacity = 20 # per location. 
        self.max_transfers = 5 # per location.
        n_max = 12

        # lambda values for poissons of returns and requests at each location
        self.lam_1_req = 3
        self.lam_2_req = 4
        self.lam_1_ret = 3
        self.lam_2_ret = 2

        # list of probabilities for above poissons
        self.probs_ret_1 = self.poisson_pmf(self.lam_1_ret, n_max)
        self.probs_req_1 = self.poisson_pmf(self.lam_1_req, n_max)
        self.probs_ret_2 = self.poisson_pmf(self.lam_2_ret, n_max)
        self.probs_req_2 = self.poisson_pmf(self.lam_2_req, n_max)
    
    def overnight_exchange(self, s, a):
        r = 0
        count_1, count_2 = s

        new_1 = count_1 - a
        new_2 = count_2 + a

        new_1 = min(max(new_1, 0), self.max_capacity)
        new_2 = min(max(new_2, 0), self.max_capacity)
        
        r += -2 * abs(a) # Handle moving cost of -$2 per car

        return new_1, new_2, r

        
    # Learned the step() is for samppling(i.e. running and agent/ simulation) not for policy iteration.
    def step(self, s, a, rng): # Step is one day from evening after moves to evening before moves
        if a not in self.actions(s):
            raise ValueError(f"Invalid action {a}. Must be one of {self.actions(s)}")
        
        new_1, new_2, r = self.overnight_exchange(s, a)

        # Handle daytime returns and requests
        req_1 = rng.poisson(self.lam_1_req)
        ret_1 = rng.poisson(self.lam_1_ret)
        req_2 = rng.poisson(self.lam_2_req)
        ret_2 = rng.poisson(self.lam_2_ret)

        req_1 = min(req_1, new_1)
        req_2 = min(req_2, new_2)

        r += 10 * (req_1 + req_2)

        new_1 = min(self.max_capacity, new_1 + ret_1 - req_1)
        new_2 = min(self.max_capacity, new_2 + ret_2 - req_2)

        s_prime = new_1, new_2
        return s_prime, r
    
    # A(s) function
    def actions(self, s):
        count_1, count_2 = s
        low = -min(count_2, self.max_transfers)
        high = min(count_1, self.max_transfers)
        return list(range(low, high + 1))
    
    # s in S. Returns all states for env
    def states(self):
        # where state = (cars at a, cars at b)
        for i in range(self.max_capacity + 1):
            for j in range(self.max_capacity + 1):
                yield(i, j)

    # Used for finding transition probabilities
    def poisson_pmf(self, lam, n_max):
        pmf = []
        for n in range(n_max):
            prob = math.exp(-lam) * (lam ** n ) / math.factorial(n)
            pmf.append(prob)

        # handle tail lumping for N-max
        pmf.append(1 - sum(pmf))

        return pmf
        

    # p(s', r | s, a)
    def transitions(self, s, a):
        n_max = 15
        # Handle deterministic state-action result portion
        new_1, new_2, r = self.overnight_exchange(s, a)


        for prob_1 in ret_1:
            for prob_2 in req_1:
                for prob_3 in ret_2:
                    for prob_4 in req_2:


        # enumerate through these lists and return prob, reward, and s'