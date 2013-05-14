class MCMCParams(object):
    def __init__(self, start, num_iterations=80000, burnin=60000):
        self.num_iterations = num_iterations
        self.burnin = burnin
        self.start = start
