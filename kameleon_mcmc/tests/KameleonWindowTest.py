"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from kameleon_mcmc.distribution.Banana import Banana
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.KameleonWindow import KameleonWindow
from numpy.core.numeric import zeros
import cProfile
import pstats


def main():
    distribution = Banana(dimension=8, bananicity=0.1, V=100.0)
    
    sigma = 5
    print "using sigma", sigma
    kernel = GaussianKernel(sigma=sigma)
    
    mcmc_sampler = KameleonWindow(distribution, kernel)
    
    start = zeros(distribution.dimension)
    mcmc_params = MCMCParams(start=start, num_iterations=80000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
#    chain.append_mcmc_output(PlottingOutput(distribution, plot_from=3000))
    chain.append_mcmc_output(StatisticsOutput(plot_times=True))
    chain.run()
    
    print distribution.emp_quantiles(chain.samples)
    
#    Visualise.visualise_distribution(distribution, chain.samples)

cProfile.run("main()", "profile.tmp")
p = pstats.Stats("profile.tmp")
p.sort_stats("cumulative").print_stats(10)
