from main.mcmc.samplers.MCMCHammer import MCMCHammer
from numpy.ma.core import reshape, arange
from random import shuffle

class MCMCHammerWindow(MCMCHammer):
    def __init__(self, distribution, kernel, nu2=0.1, gamma=0.1, sample_discard=500, num_samples_Z=1000):
        MCMCHammer.__init__(self, distribution, kernel, Z=None, nu2=nu2, gamma=gamma)
        self.sample_discard = sample_discard
        self.num_samples_Z = num_samples_Z
    
    def init(self, start):
        MCMCHammer.init(self, start)
        self.Z = reshape(start, (1, len(start)))
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "sample_discard=" + str(self.sample_discard)
        s += ", num_samples_Z=" + str(self.num_samples_Z)
        s += ", " + MCMCHammer.__str__(self)
        s += "]"
        return s
    
    def adapt(self, mcmc_chain, step_output):
        """
        Updates the sliding window of samples to use
        """
        iter_no = mcmc_chain.iteration
        samples = mcmc_chain.samples[0:(iter_no + 1)]
        
        if iter_no < self.sample_discard:
            # not necessary but for structure, its None already
            self.Z = None
        else:
            if iter_no < self.sample_discard + self.num_samples_Z:
                # use all samples after discard
                self.Z = samples[self.sample_discard:(iter_no + 1)]
            else:
                # once enough samples, use random subset without repetition
                indices = arange(self.sample_discard, (iter_no + 1))
                shuffle(indices)
                self.Z = samples[indices[0:self.num_samples_Z]]
