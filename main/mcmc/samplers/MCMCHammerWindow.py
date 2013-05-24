from main.mcmc.samplers.MCMCHammer import MCMCHammer
from numpy.ma.core import reshape, sqrt, exp, log

class MCMCHammerWindow(MCMCHammer):
    adapt_scale=False
    accstar=0.574
    sample_discard=500
    def __init__(self, distribution, kernel, nu2=0.1, gamma=0.1, window_size=5000, thinning_factor=10):
        MCMCHammer.__init__(self, distribution, kernel, Z=None, nu2=nu2, gamma=gamma)
        self.window_size = window_size
        self.thinning_factor = thinning_factor
    
    def init(self, start):
        MCMCHammer.init(self, start)
        self.Z = reshape(start, (1, len(start)))
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "window_size="+ str(self.window_size)
        s += ", thinning_factor="+ str(self.thinning_factor)
        s += ", " + MCMCHammer.__str__(self)
        s += "]"
        return s
    
    def adapt(self, mcmc_chain, step_output):
        """
        Updates the sliding window of samples to use
        """
        iter_no = mcmc_chain.iteration
        samples=mcmc_chain.samples[0:iter_no]
        
        # use samples from history window, thinned out
        if len(samples) > 0:
            sample_idxs = range(max(0, len(samples) - self.window_size + 1), \
                              len(samples), self.thinning_factor)
            self.Z = samples[sample_idxs]
        #adapt scale in the LearnScale case
        if iter_no>self.sample_discard and self.adapt_scale:
            learn_scale=1.0 / sqrt(iter_no - self.sample_discard + 1.0)
            self.nu2 = exp(log(self.nu2) + learn_scale * (exp(step_output.log_ratio) - self.accstar))
        
