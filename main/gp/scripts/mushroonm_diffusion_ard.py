"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from main.distribution.Gaussian import Gaussian
from main.experiments.SingleChainExperiment import SingleChainExperiment
from main.gp.GPData import GPData
from main.gp.mcmc.PseudoMarginalHyperparameterDistributionDiffusion import \
    PseudoMarginalHyperparameterDistributionDiffusion
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.AdaptiveMetropolisLearnScale import \
    AdaptiveMetropolisLearnScale
from main.mcmc.samplers.KameleonWindowLearnScale import KameleonWindowLearnScale
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis
from numpy.lib.twodim_base import eye
from numpy.ma.core import  ones, shape, zeros
from numpy.random import permutation, seed
import os
import sys
    
if __name__ == '__main__':
    # load data
    data,labels=GPData.get_mushroom_data()

    # throw away some data
    n=200
    seed(1)
    idx=permutation(len(data))
    idx=idx[:n]
    data=data[idx]
    labels=labels[idx]
    
    dim=shape(data)[1]

    # prior on theta and posterior target estimate
    theta_prior=Gaussian(mu=0*ones(dim), Sigma=eye(dim)*5)
    target=PseudoMarginalHyperparameterDistributionDiffusion(data, labels, \
                                                    n_importance=100, prior=theta_prior, \
                                                    ridge=1e-3)
    
    # create sampler
    burnin=1000
    num_iterations=burnin+10000
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
#     chain.append_mcmc_output(PlottingOutput(plot_from=0, lag=1))
    
    # create experiment instance to store results
    experiment_dir = str(os.path.abspath(sys.argv[0])).split(os.sep)[-1].split(".")[0] + os.sep
    experiment = SingleChainExperiment(chain, experiment_dir)
    
    experiment.run()
    sigma=GaussianKernel.get_sigma_median_heuristic(experiment.mcmc_chain.samples.T)
    print "median kernel width", sigma