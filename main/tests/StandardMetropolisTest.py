"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from main.distribution.Banana import Banana
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis
from numpy.core.numeric import zeros
import cProfile
import pstats


def main():
    distribution = Banana(dimension=8, bananicity=0.1, V=100.0)
    
    mcmc_sampler = StandardMetropolis(distribution)
    
    start=zeros(distribution.dimension)
    mcmc_params = MCMCParams(start=start, num_iterations=80000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(StatisticsOutput(plot_times=True))
    chain.run()
    
    print distribution.emp_quantiles(chain.samples)
    
cProfile.run("main()", "profile.tmp")
p = pstats.Stats("profile.tmp")
p.sort_stats("cumulative").print_stats(10)