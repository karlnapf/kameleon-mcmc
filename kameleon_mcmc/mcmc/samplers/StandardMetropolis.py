from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.mcmc.samplers.MCMCSampler import MCMCSampler
from numpy.lib.twodim_base import eye

class StandardMetropolis(MCMCSampler):
    '''
    Just a plain, old, boring Metropolis Algorithm
    with a fixed scale and a fixed covairance matrix
    '''
    is_symmetric = True
    def __init__(self, distribution, scale=None, cov=None):
        MCMCSampler.__init__(self, distribution)
        if scale is None:
            self.scale = (2.38 ** 2) / distribution.dimension
        else:
            self.scale = scale
        if cov is None:
            self.cov = eye(distribution.dimension)
        else:
            self.cov = cov
        
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "scale=" + str(self.scale)
        s += ", " + MCMCSampler.__str__(self)
        s += "]"
        return s
        
    def construct_proposal(self, y):
        return Gaussian(y, self.scale * self.cov)
    
    def adapt(self, mcmc_chain, step_output):
        """
        Nothing to be seen here, this is a nonadaptive Sampler
        """
