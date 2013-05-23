from main.distribution.Banana import Banana
from main.distribution.Gaussian import Gaussian
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.samplers.MCMCSampler import MCMCSampler
from main.tools.Visualise import Visualise
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import norm
from numpy.ma.core import array, mean

class StandardMetropolis(MCMCSampler):
    '''
    Just a plain, old, boring Metropolis Algorithm
    with an isotropic proposal and fixed scale
    '''
    is_symmetric=True
    def __init__(self,distribution,scale=None):
        MCMCSampler.__init__(self, distribution)
        if scale is None:
            self.scale=(2.38 ** 2) / distribution.dimension
        else:
            self.scale=scale
        
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "scale="+ str(self.scale)
        s += ", " + MCMCSampler.__str__(self)
        s += "]"
        return s
        
    def construct_proposal(self, y):
        return Gaussian(y,self.scale*eye(self.distribution.dimension))
    def adapt(self, mcmc_chain, step_output):
        """
        Nothing to be seen here, this is a nonadaptive Sampler
        """
        
if __name__ == '__main__':
    distribution = Banana(2)
    am_sampler = StandardMetropolis(distribution)
    
    start = array([-2.0, -2.0])
    length = 26000
    burnin = 6000
    thin = 10
    idx = range(burnin, length, thin)
    mcmc_params = MCMCParams(start=start, num_iterations=length, burnin=burnin)
    
    chain = MCMCChain(am_sampler, mcmc_params)
    chain.append_mcmc_output(ProgressOutput())
    #chain.append_mcmc_output(PlottingOutput(distribution, plot_from=6000))
    
    chain.run()
    print "\n\nStandard Metropolis:\n"
    print "    overall length: ", len(chain.samples)
    print "    burned-in length: ", len(idx)
    print "    empirical quantiles: ", distribution.emp_quantiles(chain.samples[idx])
    print "    mean: ", mean(chain.samples[idx], 0)
    print "    norm of the mean: ", norm(mean(chain.samples[idx], 0))
    print "    mean accepted rate: ", mean(chain.accepteds[idx])
    Visualise.visualise_distribution(distribution, chain.samples)