from main.mcmc.output.Output import Output
from numpy.ma.core import mean

class StatisticsOutput(Output):
    def __init__(self, print_from=1000, lag=1000):
        Output.__init__(self)
        
        self.print_from = print_from
        self.lag = lag
    
    def update(self, mcmc_chain, step_output):
        i = mcmc_chain.iteration
        if i >= self.print_from and i % self.lag == 0:
            print "mean(mcmc_chain.accepteds[0:" + str(i) + "):", \
                  mean(mcmc_chain.accepteds[0:i])
    
    def prepare(self):
        pass
