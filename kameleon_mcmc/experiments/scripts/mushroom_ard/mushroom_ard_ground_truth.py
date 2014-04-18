"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy import eye
from numpy.ma.core import ones, shape, zeros
from numpy.random import permutation
import os
import sys

from main.distribution.Gaussian import Gaussian
from main.experiments.ClusterTools import ClusterTools
from main.experiments.SingleChainExperiment import SingleChainExperiment
from main.gp.GPData import GPData
from main.gp.mcmc.PseudoMarginalHyperparameterDistributionDiffusion import PseudoMarginalHyperparameterDistributionDiffusion
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "usage:", str(sys.argv[0]).split(os.sep)[-1], "<experiment_dir_base> <number_of_experiments>"
        print "example:"
        print "python " + str(sys.argv[0]).split(os.sep)[-1] + " /nfs/nhome/live/ucabhst/kameleon_experiments/ 3"
        exit()
    
    experiment_dir_base = str(sys.argv[1])
    n = int(str(sys.argv[2]))
    
    # loop over parameters here
    
    experiment_dir = experiment_dir_base + str(os.path.abspath(sys.argv[0])).split(os.sep)[-1].split(".")[0] + os.sep
    print "running experiments", n, "times at base", experiment_dir
   
    # load data
    data,labels=GPData.get_mushroom_data()
    
    # throw away some data
    n=500
    idx=permutation(len(data))
    idx=idx[:n]
    data=data[idx]
    labels=labels[idx]
    
    dim=shape(data)[1]
    
    # prior on theta and posterior target estimate
    theta_prior=Gaussian(mu=0*ones(dim), Sigma=eye(dim)*5)
    distribution=PseudoMarginalHyperparameterDistributionDiffusion(data, labels, \
                                                    n_importance=100, prior=theta_prior, \
                                                    ridge=1e-3)

    for i in range(n):
        
        mcmc_samplers = []
        
        burnin=10000
        num_iterations=100000
        
        #mcmc_samplers.append(KameleonWindowLearnScale(distribution, kernel, stop_adapt=burnin))
        
        #mean_est = zeros(distribution.dimension, dtype="float64")
        #cov_est = 1.0 * eye(distribution.dimension)
        #cov_est[0, 0] = distribution.V
        #mcmc_samplers.append(AdaptiveMetropolisLearnScale(distribution, mean_est=mean_est, cov_est=cov_est))
        #mcmc_samplers.append(AdaptiveMetropolis(distribution, mean_est=mean_est, cov_est=cov_est))
        mcmc_samplers.append(StandardMetropolis(distribution))
        
        start = zeros(distribution.dimension, dtype="float64")
        mcmc_params = MCMCParams(start=start, num_iterations=num_iterations, burnin=burnin)
        
        mcmc_chains = [MCMCChain(mcmc_sampler, mcmc_params) for mcmc_sampler in mcmc_samplers]
        for mcmc_chain in mcmc_chains:
            mcmc_chain.append_mcmc_output(StatisticsOutput())
        
        experiments = [SingleChainExperiment(mcmc_chain, experiment_dir) for mcmc_chain in mcmc_chains]
        
        for experiment in experiments:
            ClusterTools.submit_experiment(experiment)
