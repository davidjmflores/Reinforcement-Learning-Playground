import math

# Jack's Car Rental is an environment example given on pg. 81 Example 4.2

class JacksCarRental:
    def __init__(self):
        # Jack's car rental constraints
        self.max_capacity = 20 # per location. 
        self.max_transfers = 5 # per location.
        n_max = 13 # max number of rentals and requests per day per location

        # lambda values for poissons of returns and requests at each location
        self.lam_1_req = 3
        self.lam_2_req = 4
        self.lam_1_ret = 3
        self.lam_2_ret = 2

        # list of prob mass functions for each locations returns and requests
        self.pmf_ret_1 = self.poisson_pmf(self.lam_1_ret, n_max)
        self.pmf_req_1 = self.poisson_pmf(self.lam_1_req, n_max)
        self.pmf_ret_2 = self.poisson_pmf(self.lam_2_ret, n_max)
        self.pmf_req_2 = self.poisson_pmf(self.lam_2_req, n_max)
    
    # Function handles deterministic car transfers between locations
    def overnight_exchange(self, s, a):
        r = 0
        count_1, count_2 = s

        new_1 = count_1 - a
        new_2 = count_2 + a

        new_1 = min(max(new_1, 0), self.max_capacity)
        new_2 = min(max(new_2, 0), self.max_capacity)
        
        r += -2 * abs(a) # Handle moving cost of -$2 per car

        return new_1, new_2, r

        
    # Function is for sampling algorithms.
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

        new_1 = max(0, min(self.max_capacity, new_1 + ret_1 - req_1))
        new_2 = max(0, min(self.max_capacity, new_2 + ret_2 - req_2))


        s_prime = new_1, new_2
        return s_prime, r
    
    # A(s) function
    def actions(self, s):
        count_1, count_2 = s
        low = -min(count_2, self.max_transfers)
        high = min(count_1, self.max_transfers)
        return list(range(low, high + 1))
    
    # s in S. Returns all states for env. For purpose of iteration through bellman.
    def states(self):
        # where state = (cars at a, cars at b)
        for i in range(self.max_capacity + 1):
            for j in range(self.max_capacity + 1):
                yield(i, j)

    # outputs pmf for any given poisson random variable
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
        if a not in self.actions(s):
            raise ValueError(f"Invalid action {a}. Must be one of {self.actions(s)}")
        # Handle deterministic state-action result portion
        new_1, new_2, r = self.overnight_exchange(s, a)
        
        P = {}
        R = {}

        for ret_1, p_1 in enumerate(self.pmf_ret_1):
            for ret_2, p_2 in enumerate(self.pmf_ret_2):
                for req_1, p_3 in enumerate(self.pmf_req_1):
                    for req_2, p_4 in enumerate(self.pmf_req_2):
                        temp_r = r

                        # Clamp according to problem constraints
                        rent_1 = min(req_1, new_1)
                        rent_2 = min(req_2, new_2)

                        # Calculate p, r, and s' for given configuration
                        temp_r += 10 * (rent_1 + rent_2)
                        temp_new_1 = new_1 + (ret_1 - rent_1)
                        temp_new_2 = new_2 + (ret_2 - rent_2)
                        temp_new_1 = max(0, min(self.max_capacity, temp_new_1))
                        temp_new_2 = max(0, min(self.max_capacity, temp_new_2))

                        s_prime = (temp_new_1, temp_new_2)
                        
                        p = p_1 * p_2 * p_3 * p_4
                            
                        P[s_prime] = P.get(s_prime, 0.0) + p
                        R[s_prime] = R.get(s_prime, 0.0) + p * temp_r
        
        transitions = []

        for s_prime, p_total in P.items():          # iterate only over reached next-states
            transitions.append((p_total, s_prime, R[s_prime] / p_total))
            sum(P.values)

        return transitions
