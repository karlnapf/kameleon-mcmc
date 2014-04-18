from kameleon_mcmc.mcmc.samplers.KameleonWindow import KameleonWindow
from numpy.ma.core import sqrt, exp, log

class KameleonWindowLearnScale(KameleonWindow):
    def __init__(self, distribution, kernel, nu2=0.1, gamma=None, \
                 sample_discard=500, num_samples_Z=1000, stop_adapt=20000, accstar=0.234):
        KameleonWindow.__init__(self, distribution, kernel, nu2, gamma, \
                                  sample_discard, num_samples_Z, stop_adapt)
        
        self.accstar = accstar
    
    def adapt(self, mcmc_chain, step_output):
        # this is an extension of the base adapt call
        KameleonWindow.adapt(self, mcmc_chain, step_output)
        
        iter_no = mcmc_chain.iteration
        
        if iter_no > self.sample_discard and iter_no < self.stop_adapt:
            learn_scale = 1.0 / sqrt(iter_no - self.sample_discard + 1.0)
            self.nu2 = exp(log(self.nu2) + learn_scale * (exp(step_output.log_ratio) - self.accstar))
            
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "accstar=" + str(self.accstar)
        s += ", " + KameleonWindow.__str__(self)
        s += "]"
        return s
