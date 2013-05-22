class MCMCParams(object):
    def __init__(self, start, num_iterations=80000, burnin=60000):
        self.num_iterations = num_iterations
        self.burnin = burnin
        self.start = start
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "num_iteratons="+ str(self.num_iterations)
        s += ", burnin="+ str(self.burnin)
        s += ", start="+ str(self.start)
        s += "]"
        return s