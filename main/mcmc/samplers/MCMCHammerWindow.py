from main.mcmc.samplers.MCMCHammer import MCMCHammer
from numpy.ma.extras import unique
from numpy.random import randint

class MCMCHammerWindow(MCMCHammer):
    def __init__(self, distribution, kernel, nu2=0.1, gamma=0.1, \
                 sample_discard=500, num_samples_Z=1000, stop_adapt=20000):
        
        MCMCHammer.__init__(self, distribution, kernel, Z=None, nu2=nu2, gamma=gamma)
        
        assert(stop_adapt > sample_discard)
        assert(num_samples_Z > 0)

        self.sample_discard = sample_discard
        self.num_samples_Z = num_samples_Z
        self.stop_adapt = stop_adapt
    
    def init(self, start):
        MCMCHammer.init(self, start)
        self.Z = None
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "sample_discard=" + str(self.sample_discard)
        s += ", num_samples_Z=" + str(self.num_samples_Z)
        s += ", stop_adapt=" + str(self.stop_adapt)
        s += ", " + MCMCHammer.__str__(self)
        s += "]"
        return s
    
    def adapt(self, mcmc_chain, step_output):
        """
        Updates the sliding window of samples to use
        """
        iter_no = mcmc_chain.iteration
        samples = mcmc_chain.samples[0:(iter_no + 1)]
        
        # only adapt after discard has passed
        if iter_no > self.sample_discard:
            
            if iter_no < self.sample_discard + self.num_samples_Z:
                # use all samples after discard if not yet enough
                self.Z = samples[self.sample_discard:(iter_no + 1)]
            else:
                # stop adapting at some point
                if iter_no < self.stop_adapt:
                    # once enough samples, use random subset with repetition
                    # and remove duplicates. Sampling without repetition is too expensive
                    inds = randint(iter_no - self.sample_discard, size=self.num_samples_Z) + self.sample_discard
                    unique_inds = unique(inds)
    #                print len(inds) - len(unique_inds), "collisions and", len(unique_inds), "unique samples"
                    
                    self.Z = samples[unique_inds]
