from main.distribution.Ring import Ring
from main.experiments.ClusterTools import ClusterTools
from main.experiments.SingleChainExperiment import SingleChainExperiment
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.samplers.AdaptiveMetropolis import AdaptiveMetropolis
from main.mcmc.samplers.AdaptiveMetropolisLearnScale import \
    AdaptiveMetropolisLearnScale
from main.mcmc.samplers.AdaptiveMetropolisPCA import AdaptiveMetropolisPCA
from main.mcmc.samplers.MCMCHammerWindow import MCMCHammerWindow
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis
from numpy.lib.twodim_base import eye
from numpy.ma.core import array
import os
import sys

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print "usage:", str(sys.argv[0]), "<experiment_dir> <number_of_experiments>"
        print "example:"
        print "python Ring.py /nfs/home1/ucabhst/mcmc_hammer_experiments/ 3"
        exit()
    
    experiment_dir=str(sys.argv[1])
    n=int(str(sys.argv[2]))
    
    print "running experiments", n, "times at", experiment_dir
    
    for i in range(n):
        distribution = Ring()
        
        mcmc_samplers = []
        
        kernel = GaussianKernel(sigma=1)
        mcmc_samplers.append(MCMCHammerWindow(distribution, kernel))
        
        mean_est = array([-2.0, -2.0])
        cov_est = 0.05 * eye(2)
        mcmc_samplers.append(AdaptiveMetropolis(distribution, mean_est=mean_est, cov_est=cov_est))
        mcmc_samplers.append(AdaptiveMetropolisLearnScale(distribution, mean_est=mean_est, cov_est=cov_est))
        
        num_eigen = 2
        mcmc_samplers.append(AdaptiveMetropolisPCA(distribution, num_eigen=num_eigen, mean_est=mean_est, cov_est=cov_est))
        
        mcmc_samplers.append(StandardMetropolis(distribution))
        
        start = array([-2.0, -2.0])
        mcmc_params = MCMCParams(start=start, num_iterations=50000, burnin=20000)
        
        mcmc_chains = [MCMCChain(mcmc_sampler, mcmc_params) for mcmc_sampler in mcmc_samplers]
        for mcmc_chain in mcmc_chains:
            mcmc_chain.append_mcmc_output(ProgressOutput())
        
        experiments = [SingleChainExperiment(mcmc_chain, experiment_dir) for mcmc_chain in mcmc_chains]
        
        dispatcher_filename=os.sep.join(os.path.abspath(os.path.dirname(sys.argv[0])).split(os.sep)[:-1]) + os.sep + "run_single_chain_experiment.py"
        ClusterTools.submit_experiments(experiments, dispatcher_filename)
