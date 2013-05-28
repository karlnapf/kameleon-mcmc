from main.mcmc.samplers.MCMCHammerWindow import MCMCHammerWindow
from numpy.ma.core import sqrt, exp, log

class MCMCHammerWindowLearnScale(MCMCHammerWindow):
    def __init__(self, distribution, kernel, nu2=0.1, gamma=0.1, \
                 sample_discard=500, num_samples_Z=1000, accstar=0.234):
        MCMCHammerWindow.__init__(self, distribution, kernel, nu2, gamma, sample_discard, num_samples_Z)
        
        self.accstar = accstar
    
    def adapt(self, mcmc_chain, step_output):
        # this is an extension of the base adapt call
        MCMCHammerWindow.adapt(self, mcmc_chain, step_output)
        
        iter_no = mcmc_chain.iteration
        
        if iter_no > self.sample_discard:
            learn_scale = 1.0 / sqrt(iter_no - self.sample_discard + 1.0)
            self.nu2 = exp(log(self.nu2) + learn_scale * (exp(step_output.log_ratio) - self.accstar))
            
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "accstar=" + str(self.accstar)
        s += ", " + MCMCHammerWindowLearnScale.__str__(self)
        s += "]"
        return s
