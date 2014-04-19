"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013-2014 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from numpy.lib.twodim_base import eye
from numpy.ma.core import zeros
import os
import sys

from kameleon_mcmc.distribution.Banana import Banana
from kameleon_mcmc.experiments.SingleChainExperiment import SingleChainExperiment
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.AdaptiveMetropolis import AdaptiveMetropolis
from kameleon_mcmc.mcmc.samplers.AdaptiveMetropolisLearnScale import \
    AdaptiveMetropolisLearnScale
from kameleon_mcmc.mcmc.samplers.KameleonWindowLearnScale import \
    KameleonWindowLearnScale
from kameleon_mcmc.mcmc.samplers.StandardMetropolis import StandardMetropolis


if __name__ == '__main__':
    experiment_dir = str(os.path.abspath(sys.argv[0])).split(os.sep)[-1].split(".")[0] + os.sep
    
    distribution = Banana(dimension=8, bananicity=0.1, V=100)
    sigma = GaussianKernel.get_sigma_median_heuristic(distribution.sample(1000).samples)
    sigma = 10
    print "using sigma", sigma
    kernel = GaussianKernel(sigma=sigma)
    
    
    burnin = 40000
    num_iterations = 80000
    
    #mcmc_sampler = KameleonWindowLearnScale(distribution, kernel, stop_adapt=burnin)
    mean_est = zeros(distribution.dimension, dtype="float64")
    cov_est = 1.0 * eye(distribution.dimension)
    cov_est[0, 0] = distribution.V
    mcmc_sampler = AdaptiveMetropolisLearnScale(distribution, mean_est=mean_est, cov_est=cov_est)
    #mcmc_sampler = AdaptiveMetropolis(distribution, mean_est=mean_est, cov_est=cov_est)
    #mcmc_sampler = StandardMetropolis(distribution)
        
    start = zeros(distribution.dimension, dtype="float64")
    mcmc_params = MCMCParams(start=start, num_iterations=num_iterations, burnin=burnin)
    
    mcmc_chain = MCMCChain(mcmc_sampler, mcmc_params)
    mcmc_chain.append_mcmc_output(StatisticsOutput())
    
    experiment = SingleChainExperiment(mcmc_chain, experiment_dir)
    experiment.run()
