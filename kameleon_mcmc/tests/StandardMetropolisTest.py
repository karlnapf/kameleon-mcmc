"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

import cProfile
from numpy import asarray
from numpy.core.numeric import zeros
from pickle import dump
import pstats

from main.distribution.Banana import Banana
from main.distribution.Gaussian import Gaussian
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.AdaptiveMetropolis import AdaptiveMetropolis
from main.mcmc.samplers.AdaptiveMetropolisLearnScale import AdaptiveMetropolisLearnScale
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis


def main():
    distribution = Banana(dimension=2, bananicity=0.03, V=100.0)
    
    
    mcmc_sampler = StandardMetropolis(distribution)
    
    start=zeros(distribution.dimension)
    start=asarray([0.,-2.])
    mcmc_params = MCMCParams(start=start, num_iterations=10000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(StatisticsOutput(plot_times=True, lag=1000))
#     chain.append_mcmc_output(PlottingOutput(distribution, plot_from=1, num_samples_plot=0,
#                                             colour_by_likelihood=False))
    
    chain.run()
    f=open("std_metropolis_chain_gaussian.bin", 'w')
    dump(chain, f)
    f.close()
    
    
# cProfile.run("main()", "profile.tmp")
# p = pstats.Stats("profile.tmp")
# p.sort_stats("cumulative").print_stats(10)
main()