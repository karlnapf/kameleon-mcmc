from abc import abstractmethod
from main.distribution.Gaussian import Gaussian
from main.mcmc.samplers.MCMCSampler import MCMCSampler
from numpy import eye
from numpy.ma.core import sqrt, exp, log, shape, reshape, outer, ones

class AdaptiveMetropolis(MCMCSampler):
    '''
    Plain Adaptive Metropolis by Haario et al
    adapt_scale=False uses estimated covariance * "optimal scaling"
    adapt_scale=True adapts scaling to reach "optimal acceptance rate"
    '''
    is_symmetric=True
    adapt_scale = False
    
    def __init__(self, distribution, \
                 mean_est=None, cov_est=None, \
                 sample_discard=500, sample_lag=20, accstar=0.234):
        MCMCSampler.__init__(self, distribution)
        self.globalscale = (2.38 ** 2) / distribution.dimension
        
        if mean_est is None:
            mean_est=2*ones(distribution.dimension)
            
        if cov_est is None:
            cov_est=0.05 * eye(distribution.dimension)
            
        assert (len(mean_est) == distribution.dimension)
        assert (len(cov_est) == distribution.dimension)
            
        self.mean_est = mean_est
        self.cov_est = cov_est
        self.sample_discard = sample_discard
        self.sample_lag = sample_lag
        self.accstar = accstar
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "globalscale="+ str(self.globalscale)
        s += ", sample_discard="+ str(self.sample_discard)
        s += ", sample_lag="+ str(self.sample_lag)
        s += ", accstar="+ str(self.accstar)
        s += ", " + MCMCSampler.__str__(self)
        s += "]"
        return s
    
    def mean_and_cov_adapt(self,learn_scale):
        current_1d=reshape(self.current_sample_object.samples, (self.distribution.dimension,))
        difference=current_1d - self.mean_est
        self.cov_est += learn_scale * (outer(difference, difference) - self.cov_est)
        self.mean_est += learn_scale * (current_1d - self.mean_est)
        #print "mean estimate: ", self.mean_est
    
    @abstractmethod
    def scale_adapt(self,learn_scale,step_output):
        self.globalscale = exp(log(self.globalscale) + learn_scale * (exp(step_output.log_ratio) - self.accstar))
    
    @abstractmethod
    def eigen_adapt(self):
        """
        Move along, nothing to be seen here, 
        I'm just an abstract method, and that's all I am.
        """
    
    def adapt(self, mcmc_chain, step_output):
        iter_no = mcmc_chain.iteration
        if iter_no > self.sample_discard:
            learn_scale=1.0 / sqrt(iter_no - self.sample_discard + 1.0)
            #print "current learning rate: ", learn_scale
            if self.adapt_scale:
                self.scale_adapt(learn_scale,step_output)
            if iter_no % self.sample_lag == 0:
                self.mean_and_cov_adapt(learn_scale)
                self.eigen_adapt()
    
    def construct_proposal(self, y):
        assert(len(shape(y))==1)
        return Gaussian(mu=y, Sigma=self.globalscale * self.cov_est, is_cholesky=False)
    
