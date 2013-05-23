from abc import abstractmethod
from main.distribution.Gaussian import Gaussian
from main.mcmc.samplers.MCMCSampler import MCMCSampler
from numpy import eye
from numpy.ma.core import array, sqrt, exp, log, shape, reshape, outer

class AdaptiveMetropolis(MCMCSampler):
    '''
    Plain Adaptive Metropolis by Haario et al
    adapt_scale=False uses estimated covariance * "optimal scaling"
    adapt_scale=True adapts scaling to reach "optimal acceptance rate"
    '''
    is_symmetric=True
    def __init__(self, distribution, \
                 mean_est=array([-2.0, -2.0]), cov_est=0.05 * eye(2), \
                 sample_discard=500, sample_lag=20, accstar=0.234):
        assert (len(mean_est) == distribution.dimension)
        MCMCSampler.__init__(self, distribution)
        self.globalscale = (2.38 ** 2) / distribution.dimension
        self.adapt_scale = False
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
    
#if __name__ == '__main__':
#    distribution = Banana(2)
#    length = 260000
#    burnin = 60000
#    lag = 10
#    start = array([-2.0, -2.0])
#    mean_est = array([-2.0, -2.0])
#    cov_est = 0.05 * eye(2)
#    # mcmc_sampler = MCMCHammerWindow(distribution, kernel)
#    am_sampler = AdaptiveMetropolis(distribution, adapt_scale=False, mean_est=mean_est, cov_est=cov_est)
#    mcmc_params = MCMCParams(start=start, num_iterations=length, burnin=burnin)
#    chain = MCMCChain(am_sampler, mcmc_params)
#    chain.append_mcmc_output(ProgressOutput())
#    # Xs = linspace(-20, 20, 50)
#    # Ys = linspace(-8, 20, 50)
#    chain.append_mcmc_output(PlottingOutput(distribution, plot_from=6000))
#    chain.run()
#    idx = range(burnin, length, lag)
#    print distribution.emp_quantiles(chain.samples[idx])
#    
    # Visualise.visualise_distribution(distribution, chain.samples)
