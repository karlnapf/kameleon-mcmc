from main.mcmc.samplers.MCMCSampler import MCMCSampler
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis


class GMMMetropolis(StandardMetropolis):
    '''
    Runs StandardMetropolis for a number of iterations, performs a couple of
    EM instances to fit a Gaussian Mixture Model which is subsequently used
    as a static proposal distribution
    '''
    
    def __init__(self, distribution):
        MCMCSampler.__init__(self, distribution)
    
    def __str__(self):
        raise NotImplementedError()
    
    def adapt(self, mcmc_chain, step_output):
        raise NotImplementedError()
    
    def construct_proposal(self, y):
        raise NotImplementedError()
    
