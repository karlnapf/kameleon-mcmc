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
from kameleon_mcmc.gp.mcmc.PseudoMarginalHyperparameterDistributionDiffusion import \
    PseudoMarginalHyperparameterDistributionDiffusion
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.PlottingOutput import PlottingOutput
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.AdaptiveMetropolisLearnScale import \
    AdaptiveMetropolisLearnScale
from kameleon_mcmc.mcmc.samplers.KameleonWindowLearnScale import KameleonWindowLearnScale
from kameleon_mcmc.mcmc.samplers.StandardMetropolis import StandardMetropolis
from numpy.lib.twodim_base import eye
from numpy.ma.core import  ones, shape, zeros
from numpy.random import permutation, seed
import os
import sys
    
if __name__ == '__main__':
    # load data
    data,labels=GPData.get_mushroom_data()

    # throw away some data
    n=1000
    #seed(1)
    idx=permutation(len(data))
    idx=idx[:n]
    idx=range(n)
    data=data[idx]
    labels=labels[idx]
    
    dim=shape(data)[1]

    # prior on theta and posterior target estimate
    theta_prior=Gaussian(mu=0*ones(dim), Sigma=eye(dim)*5)
    target=PseudoMarginalHyperparameterDistributionDiffusion(data, labels, \
                                                    n_importance=100, prior=theta_prior, \
                                                    ridge=1e-3)
    
    # create sampler
    burnin=50000
    num_iterations=burnin+500000
    kernel = GaussianKernel(sigma=23.0)
#     sampler=KameleonWindowLearnScale(target, kernel, stop_adapt=burnin)
#    sampler=AdaptiveMetropolisLearnScale(target)
    sampler=StandardMetropolis(target)
    
    # posterior mode derived by initial tests
    start=zeros(target.dimension)
    params = MCMCParams(start=start, num_iterations=num_iterations, burnin=burnin)
    
    # create MCMC chain
    chain=MCMCChain(sampler, params)
    chain.append_mcmc_output(StatisticsOutput(print_from=0, lag=100))
    #chain.append_mcmc_output(PlottingOutput(plot_from=0, lag=100))
    
    # create experiment instance to store results
    experiment_dir = str(os.path.abspath(sys.argv[0])).split(os.sep)[-1].split(".")[0] + os.sep
    experiment = SingleChainExperiment(chain, experiment_dir)
    
    experiment.run()
    sigma=GaussianKernel.get_sigma_median_heuristic(experiment.mcmc_chain.samples.T)
    print "median kernel width", sigma
