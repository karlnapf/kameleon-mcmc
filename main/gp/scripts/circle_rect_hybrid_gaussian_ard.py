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
from main.gp.mcmc.PseudoMarginalHyperparameterDistribution import \
    PseudoMarginalHyperparameterDistribution
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.AdaptiveMetropolisLearnScale import \
    AdaptiveMetropolisLearnScale
from main.mcmc.samplers.KameleonWindowLearnScale import KameleonWindowLearnScale
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis
from matplotlib.pyplot import plot
from numpy.lib.twodim_base import eye
from numpy.ma.core import mean, std, ones, shape
from numpy.ma.extras import vstack, hstack
import os
import sys
    
if __name__ == '__main__':
    # sample data
    data_circle, labels_circle=GPData.sample_circle_data(n=40, seed_init=1)
    data_rect, labels_rect=GPData.sample_rectangle_data(n=60, seed_init=1)
    
    # combine
    data=vstack((data_circle, data_rect))
    labels=hstack((labels_circle, labels_rect))
    dim=shape(data)[1]
    
    # normalise data
    data-=mean(data, 0)
    data/=std(data,0)

    # plot
    idx_a=labels>0
    idx_b=labels<0
    plot(data[idx_a,0], data[idx_a,1],"ro")
    plot(data[idx_b,0], data[idx_b,1],"bo")
    
    # prior on theta and posterior target estimate
    theta_prior=Gaussian(mu=0*ones(dim), Sigma=eye(dim)*5)
    target=PseudoMarginalHyperparameterDistribution(data, labels, \
                                                    n_importance=100, prior=theta_prior, \
                                                    ridge=1e-3)
    
    # create sampler
    burnin=10000
    num_iterations=burnin+300000
    kernel = GaussianKernel(sigma=5.0)
    sampler=KameleonWindowLearnScale(target, kernel, stop_adapt=burnin)
#    sampler=AdaptiveMetropolisLearnScale(target)
#    sampler=StandardMetropolis(target)
    
    start=0.0*ones(target.dimension)
    params = MCMCParams(start=start, num_iterations=num_iterations, burnin=burnin)
    
    # create MCMC chain
    chain=MCMCChain(sampler, params)
    chain.append_mcmc_output(StatisticsOutput(print_from=0, lag=100))
    #chain.append_mcmc_output(PlottingOutput(plot_from=0, lag=500))
    
    # create experiment instance to store results
    experiment_dir = str(os.path.abspath(sys.argv[0])).split(os.sep)[-1].split(".")[0] + os.sep
    experiment = SingleChainExperiment(chain, experiment_dir)
    
    experiment.run()