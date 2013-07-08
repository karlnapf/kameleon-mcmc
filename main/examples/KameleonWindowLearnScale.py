"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from main.distribution.Banana import Banana
from main.distribution.Flower import Flower
from main.distribution.Ring import Ring
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.KameleonWindowLearnScale import KameleonWindowLearnScale
from main.tools.Visualise import Visualise
from numpy.core.numeric import inf
from numpy.ma.core import asarray


def main():
    # define the MCMC target distribution
    # possible distributions are in main.distribution: Banana, Flower, Ring
#    distribution = Banana(dimension=2, bananicity=0.03, V=100.0)
    distribution = Ring()
    
    # create instance of kameleon sampler that learns the length scale
    # can be replaced by any other sampler in main.mcmc.samplers
    kernel = GaussianKernel(sigma=5)
    mcmc_sampler = KameleonWindowLearnScale(distribution, kernel, stop_adapt=inf, nu2=0.05)
    
    # mcmc chain and its parameters
    start = asarray([0,-3])
    mcmc_params = MCMCParams(start=start, num_iterations=30000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    # plot every iteration and print some statistics
    chain.append_mcmc_output(PlottingOutput(distribution, plot_from=2000))
    chain.append_mcmc_output(StatisticsOutput())
    
    # run cmcm
    chain.run()
    
    # print empirical quantiles
    burnin=10000
    print distribution.emp_quantiles(chain.samples[burnin:])
    
    Visualise.visualise_distribution(distribution, chain.samples)

if __name__ == "__main__":
    main()
