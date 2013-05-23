from main.mcmc.samplers.AdaptiveMetropolis import AdaptiveMetropolis
from numpy import eye
from numpy.ma.core import array

class AdaptiveMetropolisLearnScale(AdaptiveMetropolis):
    '''
    Plain Adaptive Metropolis by Haario et al
    adapt_scale=True adapts scaling to reach "optimal acceptance rate"
    '''
    is_symmetric=True
    adapt_scale = True
    
    def __init__(self, distribution, \
                 mean_est=array([-2.0, -2.0]), cov_est=0.05 * eye(2), \
                 sample_discard=500, sample_lag=20, accstar=0.234):
        AdaptiveMetropolis.__init__(self, distribution, mean_est, cov_est, \
                                    sample_discard, sample_lag, accstar)
        
