"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.experiments.SingleChainExperiment import SingleChainExperiment
from kameleon_mcmc.gp.GPData import GPData
from kameleon_mcmc.gp.mcmc.PseudoMarginalHyperparameterDistribution import PseudoMarginalHyperparameterDistribution
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.PlottingOutput import PlottingOutput
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.AdaptiveMetropolisLearnScale import AdaptiveMetropolisLearnScale
from kameleon_mcmc.mcmc.samplers.AdaptiveMetropolis import AdaptiveMetropolis
from kameleon_mcmc.mcmc.samplers.KameleonWindowLearnScale import KameleonWindowLearnScale
from kameleon_mcmc.mcmc.samplers.StandardMetropolis import StandardMetropolis
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import cholesky
from numpy.ma.core import mean, ones, shape, asarray, zeros
from numpy.ma.extras import cov
from numpy.random import permutation, seed
from scipy.linalg.basic import solve_triangular
from kameleon_mcmc.experiments.ClusterTools import ClusterTools
import os
import sys

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
    data,labels=GPData.get_glass_data()

    # normalise and whiten dataset
    data-=mean(data, 0)
    L=cholesky(cov(data.T))
    data=solve_triangular(L, data.T, lower=True).T
    dim=shape(data)[1]

    # prior on theta and posterior target estimate
    theta_prior=Gaussian(mu=0*ones(dim), Sigma=eye(dim)*5)
    distribution=PseudoMarginalHyperparameterDistribution(data, labels, \
                                                    n_importance=100, prior=theta_prior, \
                                                    ridge=1e-3)

    sigma = 23.0
    print "using sigma", sigma
    kernel = GaussianKernel(sigma=sigma)
    
    for i in range(n):
        
        mcmc_samplers = []
        
        burnin=20000
        num_iterations=100000
        
        mcmc_samplers.append(KameleonWindowLearnScale(distribution, kernel, stop_adapt=burnin))
        
        mean_est = zeros(distribution.dimension, dtype="float64")
        cov_est = 1.0 * eye(distribution.dimension)
        #cov_est[0, 0] = distribution.V
        mcmc_samplers.append(AdaptiveMetropolisLearnScale(distribution, mean_est=mean_est, cov_est=cov_est))
        mcmc_samplers.append(AdaptiveMetropolis(distribution, mean_est=mean_est, cov_est=cov_est))
        #mcmc_samplers.append(StandardMetropolis(distribution))
        
        start = zeros(distribution.dimension, dtype="float64")
        mcmc_params = MCMCParams(start=start, num_iterations=num_iterations, burnin=burnin)
        
        mcmc_chains = [MCMCChain(mcmc_sampler, mcmc_params) for mcmc_sampler in mcmc_samplers]
        for mcmc_chain in mcmc_chains:
            mcmc_chain.append_mcmc_output(StatisticsOutput())
        
        experiments = [SingleChainExperiment(mcmc_chain, experiment_dir) for mcmc_chain in mcmc_chains]
        
        for experiment in experiments:
            ClusterTools.submit_experiment(experiment)
