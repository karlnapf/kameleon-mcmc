from main.distribution.Banana import Banana
from main.distribution.Flower import Flower
from main.distribution.Gaussian import Gaussian
from main.distribution.Ring import Ring
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.samplers.AdaptiveMetropolis import AdaptiveMetropolis
from main.mcmc.samplers.MCMCHammerWindow import MCMCHammerWindow
from main.mcmc.samplers.MCMCSampler import MCMCSampler
from main.tools.Visualise import Visualise
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import norm
from numpy.ma.core import array, mean, zeros

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
        
if __name__ == '__main__':
    banana = Banana(dimension=8, bananicity=0.1)
    Sigma = eye(banana.dimension)
    Sigma[0, 0] = banana.V
    
    m_sampler = StandardMetropolis(banana, cov=Sigma)
    am_sampler = AdaptiveMetropolis(banana, adapt_scale=True, mean_est=zeros([banana.dimension]), cov_est=Sigma)
    kernel = GaussianKernel(sigma=1)
    kameleon_sampler = MCMCHammerWindow(banana, kernel)
    
    start = zeros([banana.dimension])
    length = 8000
    burnin = 4000
    thin = 1
    idx = range(burnin, length, thin)
    mcmc_params = MCMCParams(start=start, num_iterations=length, burnin=burnin)
    
    chain1 = MCMCChain(m_sampler, mcmc_params)
    chain1.append_mcmc_output(ProgressOutput())
    
    chain2 = MCMCChain(am_sampler, mcmc_params)
    chain2.append_mcmc_output(ProgressOutput())
    
    chain3 = MCMCChain(kameleon_sampler, mcmc_params)
    chain3.append_mcmc_output(ProgressOutput())
    
    
    chain1.run()
    chain2.run()
    chain3.run()
    print "\n\nStandard Metropolis:\n"
    print "    overall length: ", len(chain1.samples)
    print "    burned-in length: ", len(idx)
    print "    empirical quantiles: ", banana.emp_quantiles(chain1.samples[idx])
    print "    mean: ", mean(chain1.samples[idx], 0)
    print "    norm of the mean: ", norm(mean(chain1.samples[idx], 0))
    print "    mean accepted rate: ", mean(chain1.accepteds[idx])
    
    print "\n\nAdaptive Metropolis - learned scale:\n"
    print "    overall length: ", len(chain2.samples)
    print "    burned-in length: ", len(idx)
    print "    empirical quantiles: ", banana.emp_quantiles(chain2.samples[idx])
    print "    mean: ", mean(chain2.samples[idx], 0)
    print "    norm of the mean: ", norm(mean(chain2.samples[idx], 0))
    print "    mean accepted rate: ", mean(chain2.accepteds[idx])
    
    print "\n\nKameleon sampler:\n"
    print "    overall length: ", len(chain3.samples)
    print "    burned-in length: ", len(idx)
    print "    empirical quantiles: ", banana.emp_quantiles(chain3.samples[idx])
    print "    mean: ", mean(chain3.samples[idx], 0)
    print "    norm of the mean: ", norm(mean(chain3.samples[idx], 0))
    print "    mean accepted rate: ", mean(chain3.accepteds[idx])
    # Visualise.visualise_distribution(banana, chain.samples)
