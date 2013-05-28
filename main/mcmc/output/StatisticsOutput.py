from main.mcmc.output.Output import Output
from numpy.ma.core import mean, array
import time

class StatisticsOutput(Output):
    def __init__(self, print_from=1000, lag=1000):
        Output.__init__(self)
        
        self.print_from = print_from
        self.lag = lag
        
        # for time estimates
        self.times = []
        self.times.append(time.time())
    
    def update(self, mcmc_chain, step_output):
        i = mcmc_chain.iteration
        if i >= self.print_from and i % self.lag == 0:
            self.times.append(time.time())
            
            print "mean(mcmc_chain.accepteds[0:" + str(i) + "]:", \
                  mean(mcmc_chain.accepteds[0:i])
            
            print int(self.get_percent_done(i, mcmc_chain.mcmc_params.num_iterations)), \
            "\tpercent done in ", int(round(sum(array(self.times) - self.times[0]))), \
            "seconds. ETA:", self.get_estimated_time_remaining(i, mcmc_chain.mcmc_params.num_iterations)
         
    def prepare(self):
        pass

    def get_percent_done(self, iteration, num_iterations):
        return float(iteration) / num_iterations * 100

    def get_estimated_time_remaining(self, iteration, num_iterations):
        percent_done = self.get_percent_done(iteration, num_iterations)
        eta = sum(array(self.times) - self.times[0]) / percent_done * (100 - percent_done)
        return int(round(eta))
