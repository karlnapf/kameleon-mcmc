from main.distribution.Banana import Banana
from main.experiments.ClusterTools import ClusterTools
from main.experiments.SingleChainExperiment import SingleChainExperiment
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.samplers.AdaptiveMetropolisLearnScale import \
    AdaptiveMetropolisLearnScale
from main.mcmc.samplers.AdaptiveMetropolisPCA import AdaptiveMetropolisPCA
from main.mcmc.samplers.MCMCHammerWindowLearnScale import \
    MCMCHammerWindowLearnScale
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis
from numpy.lib.twodim_base import eye
from numpy.ma.core import zeros
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "usage:", str(sys.argv[0]).split(os.sep)[-1], "<experiment_dir_base> <number_of_experiments>"
        print "example:"
        print "python " + str(sys.argv[0]).split(os.sep)[-1] + " /nfs/home1/ucabhst/mcmc_hammer_experiments/ 3"
        exit()
    
    experiment_dir_base = str(sys.argv[1])
    n = int(str(sys.argv[2]))
    
    # loop over parameters here
    
    experiment_dir = experiment_dir_base + str(os.path.abspath(sys.argv[0])).split(os.sep)[-1].split(".")[0] + os.sep
    print "running experiments", n, "times at base", experiment_dir
    
    distribution = Banana(dimension=8, bananicity=0.1, V=100)
    sigma = GaussianKernel.get_sigma_median_heuristic(distribution.sample(1000).samples)
    print "using sigma", sigma
    kernel = GaussianKernel(sigma=sigma)
    
    for i in range(n):
        
        mcmc_samplers = []
        
        # median heurist: pairwise distances
        
        mcmc_samplers.append(MCMCHammerWindowLearnScale(distribution, kernel))
        
        mean_est = zeros(distribution.dimension, dtype="float64")
        cov_est = 1.0 * eye(distribution.dimension)
        cov_est[0, 0] = distribution.V
        mcmc_samplers.append(AdaptiveMetropolisLearnScale(distribution, mean_est=mean_est, cov_est=cov_est))
        
        num_eigen = distribution.dimension
        mcmc_samplers.append(AdaptiveMetropolisPCA(distribution, num_eigen=num_eigen, mean_est=mean_est, cov_est=cov_est))
        
        mcmc_samplers.append(StandardMetropolis(distribution))
        
        start = zeros(distribution.dimension, dtype="float64")
        mcmc_params = MCMCParams(start=start, num_iterations=80000, burnin=40000)
        
        mcmc_chains = [MCMCChain(mcmc_sampler, mcmc_params) for mcmc_sampler in mcmc_samplers]
        for mcmc_chain in mcmc_chains:
            mcmc_chain.append_mcmc_output(ProgressOutput())
        
        experiments = [SingleChainExperiment(mcmc_chain, experiment_dir) for mcmc_chain in mcmc_chains]
        
        for experiment in experiments:
            ClusterTools.submit_experiment(experiment)
