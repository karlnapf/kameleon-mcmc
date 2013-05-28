from main.mcmc.output.Output import Output
from numpy.ma.core import mean
import time

class StatisticsOutput(Output):
    def __init__(self, print_from=1000, lag=1000):
        Output.__init__(self)
        
        self.print_from = print_from
        self.lag = lag
        
        # for every update, append the time needed for it
        self.start_time = time.time()
        self.times = []
        self.times.append(0)
    
    def update(self, mcmc_chain, step_output):
        i = mcmc_chain.iteration
        if i >= self.print_from and i % self.lag == 0:
            self.times.append(time.time() - self.times[-1] - self.start_time)
                
            print "iteration:", i
            print "mean acceptance:", mean(mcmc_chain.accepteds[0:i])
            
            elapsed=int(round(sum(self.times)))
            percent=int(self.get_percent_done(i, mcmc_chain.mcmc_params.num_iterations))
            since_last=int(round(self.times[-1]))
            remaining=self.get_estimated_time_remaining(i, mcmc_chain.mcmc_params.num_iterations)
            total=elapsed+remaining
            
            print percent, "percent done in ", elapsed, "seconds"
            print "Since last update:", since_last, "seconds"
            print "remaining (estimated):", remaining, "seconds"
            print "total (estimated):", total, "seconds"
            print ""
            
            print "times between updates (not cumulative):", self.times
         
    def prepare(self):
        pass

    def get_percent_done(self, iteration, num_iterations):
        return float(iteration + 1) / num_iterations * 100

    def get_estimated_time_remaining(self, iteration, num_iterations):
        percent_done = self.get_percent_done(iteration, num_iterations)
        eta = sum(self.times) / percent_done * (100.0 - percent_done)
        return int(round(eta))
