# Realized this problem is a lot harder than previously assumed. Transitions are now stochastic
# meaning I really have to do some work in my transitions function. Additionally, adding poisson
# creates technical details I need to look into

class JacksCarRental:
    def __init__(self, policy, gamma=0.9):
        self.max_cars = 20 # per location. So 40 total
        self.max_cars_moved = 5 # per location. So 10 total
        self.lambda_a_req = 3
        self.lambda_b_req = 4
        self.lambda_a_ret = 3
        self.lambda_b_ret = 2
        self.actions = (0,1,2,3,4,5,6,7,8,9,10) # can make nicer. like list 0 through (2*self.max_cars_moved)
        # maybe I should make actions a function where I add the two entries and return the net but self.actions
        # could be self.actions = {(0,1),(0,2),(0,3),(0,4),(0,5),...} for all actions less than 10
    def poisson(self, n, lambda):
        #prob = ((exp(lambda, n) / factorial(n)) * exp(-lambda))
        #return prob
    
    def step(self, s, a): # One day
        if a not in self.actions:
            raise ValueError(f"Invalid action {a}. Must be one of {self.actions}")
        # Do I handle making sure actions dont surpass 10 total here? or in transitions?
        if a[0] + a[1] > self.max_cars_moved:
            raise ValueError(f"Transferred too many cars overnight!")
        
        s_prime = s
        r = 0

        # Handle overnight actions
        # actions from b will impact a
        if s_prime[0] + a[1] > self.max_cars: # move cars from b to a
            b_moved = self.max_cars - s_prime[0]
            s_prime[0] += b_moved
            r += -2 * b_moved
        else: 
            s_prime[0] += a[1]
            r += -2 * a[1]
        if s_prime[1] + a[0] > self.max_cars: # move cars from a to b
            a_moved = self.max_cars - s_prime[1]
            s_prime[1] += a_moved
            r += -2 * a_moved
        else: 
            s_prime[1] += a[0]
            r += -2 * a[0]
        
        # Handle daytime returns and requests

        # sample requests and returns for a and b
        # a_req = self.poisson
        # a_ret = self.poisson
        # b_req = self.poisson
        # b_ret = self.poisson
        s_prime[0] += a_ret
        if s_prime[0] - a_req < 0: a_req += s_prime[0] - a_req
        s_prime[1] += b_ret
        if s_prime[1] - b_req < 0: b_req += s_prime[1] - b_req
        r += 10 * (a_req + b_req) # +$2 for every car requested at each location

        return s_prime, r

        # now that we have current state at end of day, we take actions
    def states(self):
        # where state = (cars at a, cars at b)
        for i in range(20):
            for j in range(20):
                yield(i, j)
    
    def transitions(self, s, a):
        s_prime, r = self.step(s, a)
        return [(1.0, s_prime, r)]