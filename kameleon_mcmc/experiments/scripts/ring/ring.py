from kameleon_mcmc.distribution.Ring import Ring
from kameleon_mcmc.experiments.ClusterTools import ClusterTools
from kameleon_mcmc.experiments.SingleChainExperiment import SingleChainExperiment
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.AdaptiveMetropolisLearnScale import \
    AdaptiveMetropolisLearnScale
from kameleon_mcmc.mcmc.samplers.AdaptiveMetropolisPCA import AdaptiveMetropolisPCA
from kameleon_mcmc.mcmc.samplers.KameleonWindowLearnScale import \
    KameleonWindowLearnScale
from kameleon_mcmc.mcmc.samplers.StandardMetropolis import StandardMetropolis
from numpy.lib.twodim_base import eye
from numpy.ma.core import array
import os
import sys

if __name__ == '__main__':
    if len(sys.argv)!=5:
        print "usage:", str(sys.argv[0]).split(os.sep)[-1], "<experiment_dir_base> <number_of_experiments> <number_of_iterations> <burnin>"
        print "example:"
        print "python " + str(sys.argv[0]).split(os.sep)[-1] + " /nfs/home1/ucabhst/kameleon_experiments 3 5000 2000"
        exit()
    
    experiment_dir_base=str(sys.argv[1])
    n=int(str(sys.argv[2]))
    num_iterations=int(str(sys.argv[3]))
    burnin=int(str(sys.argv[4]))
    
    # loop over parameters here
    
    experiment_dir=experiment_dir_base +os.sep + str(os.path.abspath(sys.argv[0])).split(os.sep)[-1].split(".")[0] + os.sep
    print "running experiments", n, "times at base", experiment_dir
    
    for i in range(n):
        distribution = Ring()
        
        mcmc_samplers = []
        
        kernel = GaussianKernel(sigma=1)
#        mcmc_samplers.append(KameleonWindow(distribution, kernel))
        mcmc_samplers.append(KameleonWindowLearnScale(distribution, kernel))
        
        mean_est = array([-2.0, -2.0])
        cov_est = 0.05 * eye(2)
#        mcmc_samplers.append(AdaptiveMetropolis(distribution, mean_est=mean_est, cov_est=cov_est))
        mcmc_samplers.append(AdaptiveMetropolisLearnScale(distribution, mean_est=mean_est, cov_est=cov_est))
        
        num_eigen = 2
        mcmc_samplers.append(AdaptiveMetropolisPCA(distribution, num_eigen=num_eigen, mean_est=mean_est, cov_est=cov_est))
        
        mcmc_samplers.append(StandardMetropolis(distribution))
        
        start = array([-2.0, -2.0])
        mcmc_params = MCMCParams(start=start, num_iterations=num_iterations, burnin=burnin)
        
        mcmc_chains = [MCMCChain(mcmc_sampler, mcmc_params) for mcmc_sampler in mcmc_samplers]
        for mcmc_chain in mcmc_chains:
            mcmc_chain.append_mcmc_output(StatisticsOutput())
        
        experiments = [SingleChainExperiment(mcmc_chain, experiment_dir) for mcmc_chain in mcmc_chains]
        
        for experiment in experiments:
            ClusterTools.submit_experiment(experiment)
