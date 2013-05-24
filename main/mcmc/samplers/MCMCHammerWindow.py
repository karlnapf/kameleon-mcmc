from main.mcmc.samplers.MCMCHammer import MCMCHammer
from numpy.ma.core import reshape

class MCMCHammerWindow(MCMCHammer):
    def __init__(self, distribution, kernel, eta=2.38, gamma=0.1, window_size=5000, thinning_factor=10):
        MCMCHammer.__init__(self, distribution, kernel, Z=None, eta=0.1, gamma=0.1)
        len
        self.kernel = kernel
        self.eta = eta
        self.gamma = gamma
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
        samples=mcmc_chain.samples[0:mcmc_chain.iteration]
        
        # use samples from history window, thinned out
        if len(samples) > 0:
            sample_idxs = range(max(0, len(samples) - self.window_size + 1), \
                              len(samples), self.thinning_factor)
            self.Z = samples[sample_idxs]
        
