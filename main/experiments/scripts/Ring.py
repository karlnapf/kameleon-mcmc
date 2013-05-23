from posixpath import expanduser
import os
import sys
to_add=os.sep.join(os.path.abspath(os.path.dirname(sys.argv[0])).split(os.sep)[0:-3])
sys.path.append(to_add)

from main.distribution.Ring import Ring
from main.experiments.SingleChainExperiment import SingleChainExperiment
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.samplers.AdaptiveMetropolis import AdaptiveMetropolis
from main.mcmc.samplers.AdaptiveMetropolisPCA import AdaptiveMetropolisPCA
from main.mcmc.samplers.MCMCHammerWindow import MCMCHammerWindow
from main.tools.ClusterTools import ClusterTools
from numpy.lib.twodim_base import eye
from numpy.ma.core import array

if __name__ == '__main__':
    distribution = Ring()
    
    mcmc_samplers = []
    
    kernel = GaussianKernel(sigma=1)
    mcmc_samplers.append(MCMCHammerWindow(distribution, kernel))
    
    mean_est = array([-2.0, -2.0])
    cov_est = 0.05 * eye(2)
    mcmc_samplers.append(AdaptiveMetropolis(distribution, adapt_scale=True, mean_est=mean_est, cov_est=cov_est))
    
    num_eigen = 2
    mcmc_samplers.append(AdaptiveMetropolisPCA(distribution, num_eigen=num_eigen, mean_est=mean_est, cov_est=cov_est))
    
    start = array([-2.0, -2.0])
    mcmc_params = MCMCParams(start=start, num_iterations=200, burnin=50)
    
    mcmc_chains = [MCMCChain(mcmc_sampler, mcmc_params) for mcmc_sampler in mcmc_samplers]
    experiment_dir = expanduser("~") + os.sep + "mcmc_hammer_experiments" + os.sep
    experiments = [SingleChainExperiment(mcmc_chain, folder_prefix=experiment_dir) for mcmc_chain in mcmc_chains]
    
    ClusterTools.submit_experiments(experiments)
