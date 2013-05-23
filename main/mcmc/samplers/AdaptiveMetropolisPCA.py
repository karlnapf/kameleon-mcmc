from main.distribution.Banana import Banana
from main.distribution.Discrete import Discrete
from main.distribution.Gaussian import Gaussian
from main.distribution.MixtureDistribution import MixtureDistribution
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.samplers.AdaptiveMetropolis import AdaptiveMetropolis
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import svd, norm
from numpy.ma.core import array, sqrt, ones, shape, exp, log, reshape, mean

class AdaptiveMetropolisPCA(AdaptiveMetropolis):
    '''
    Adaptive Metropolis with adaptive eigen-directionwise scaling
    '''
    def __init__(self, distribution, num_eigen=2, \
                 mean_est=array([-2.0, -2.0]), cov_est=0.05 * eye(2), \
                 sample_discard=500, sample_lag=10, accstar=0.234):
        AdaptiveMetropolis.__init__(self, distribution=distribution, adapt_scale=True, \
                                     mean_est=mean_est, cov_est=cov_est, \
                                     sample_discard=sample_discard, sample_lag=sample_lag, accstar=accstar)
        assert (num_eigen <= distribution.dimension)
        self.num_eigen = num_eigen
        self.dwscale = self.globalscale * ones([self.num_eigen])
        u, s, _ = svd(self.cov_est)
        self.eigvalues = s[0:self.num_eigen]
        self.eigvectors = u[:, 0:self.num_eigen]
        
        
    def construct_proposal(self, y):
        assert(len(shape(y)) == 1)
        m = MixtureDistribution(self.distribution.dimension, self.num_eigen)
        m.mixing_proportion = Discrete((self.eigvalues + 1) / (sum(self.eigvalues) + self.num_eigen))
        # print "current mixing proportion: ", m.mixing_proportion.omega
        for ii in range(self.num_eigen):
            L = sqrt(self.dwscale[ii] * self.eigvalues[ii]) * reshape(self.eigvectors[:, ii], (self.distribution.dimension, 1))
            m.components[ii] = Gaussian(y, L, is_cholesky=True, ell=1)
        # Z=m.sample(1000).samples
        # Visualise.plot_data(Z)
        return m
    
    def eigen_adapt(self):
        u, s, _ = svd(self.cov_est)
        self.eigvalues = s[0:self.num_eigen]
        # print "estimated eigenvalues: ", self.eigvalues
        self.eigvectors = u[:, 0:self.num_eigen]
        
    def scale_adapt(self, learn_scale, step_output):
        which_component = step_output.sample.which_component
        self.dwscale[which_component] = exp(log(self.dwscale[which_component]) + learn_scale * (exp(step_output.log_ratio) - self.accstar))
        # print "current learned scales: ", self.dwscale

if __name__ == '__main__':
    
    distribution = Banana(2)
    NumEigen = 2
    mean_est = array([-2.0, -2.0])
    cov_est = 0.05 * eye(2)
    """define my samplers"""
    am_sampler1 = AdaptiveMetropolis(distribution, adapt_scale=False, mean_est=mean_est, cov_est=cov_est)
    am_sampler2 = AdaptiveMetropolis(distribution, adapt_scale=True, mean_est=mean_est, cov_est=cov_est)
    am_pca_sampler = AdaptiveMetropolisPCA(distribution, NumEigen=NumEigen, mean_est=mean_est, cov_est=cov_est)
    
    start = array([-2.0, -2.0])
    length = 260000
    burnin = 60000
    thin = 10
    idx = range(burnin, length, thin)
    
    mcmc_params = MCMCParams(start=start, num_iterations=length, burnin=burnin)
    
    """define my chains and append the progress output"""
    chain1 = MCMCChain(am_sampler1, mcmc_params)
    chain2 = MCMCChain(am_sampler2, mcmc_params)
    chain3 = MCMCChain(am_pca_sampler, mcmc_params)
    
    chain1.append_mcmc_output(ProgressOutput())
    chain2.append_mcmc_output(ProgressOutput())
    chain3.append_mcmc_output(ProgressOutput())
    # chain.append_mcmc_output(PlottingOutput(distribution, plot_from=inf))
    """run them!"""
    chain1.run()
    chain2.run()
    chain3.run()
    
    
    print "\n\nAdaptive Metropolis (fixed global scaling):\n"
    print "    overall length: ", len(chain1.samples)
    print "    burned-in length: ", len(idx)
    print "    empirical quantiles: ", distribution.emp_quantiles(chain1.samples[idx])
    print "    mean: ", mean(chain1.samples[idx], 0)
    print "    norm of the mean: ", norm(mean(chain1.samples[idx], 0))
    print "    mean accepted rate: ", mean(chain1.accepteds[idx])
    
    print "\n\nAdaptive Metropolis (learned global scale):\n"
    print "    overall length: ", len(chain2.samples)
    print "    burned-in length: ", len(idx)
    print "    empirical quantiles: ", distribution.emp_quantiles(chain2.samples[idx])
    print "    mean: ", mean(chain2.samples[idx], 0)
    print "    norm of the mean: ", norm(mean(chain2.samples[idx], 0))
    print "    mean accepted rate: ", mean(chain2.accepteds[idx])
    
    print "\n\nAdaptive Metropolis PCA (learned directionwise scaling):\n"
    print "    overall length: ", len(chain3.samples)
    print "    burned-in length: ", len(idx)
    print "    empirical quantiles: ", distribution.emp_quantiles(chain3.samples[idx])
    print "    mean: ", mean(chain3.samples[idx], 0)
    print "    norm of the mean: ", norm(mean(chain3.samples[idx], 0))
    print "    mean accepted rate: ", mean(chain3.accepteds[idx])
    
    # Visualise.visualise_distribution(distribution, chain1.samples[idx])
    
    
