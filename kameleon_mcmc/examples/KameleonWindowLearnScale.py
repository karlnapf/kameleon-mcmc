"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from kameleon_mcmc.distribution.Banana import Banana
from kameleon_mcmc.distribution.Flower import Flower
from kameleon_mcmc.distribution.Ring import Ring
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.PlottingOutput import PlottingOutput
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.KameleonWindowLearnScale import KameleonWindowLearnScale
from kameleon_mcmc.tools.Visualise import Visualise
from numpy.core.numeric import inf
from numpy.ma.core import asarray


def main():
    # define the MCMC target distribution
    # possible distributions are in kameleon_mcmc.distribution: Banana, Flower, Ring
#    distribution = Banana(dimension=2, bananicity=0.03, V=100.0)
    distribution = Ring()
    
    # create instance of kameleon sampler that learns the length scale
    # can be replaced by any other sampler in kameleon_mcmc.mcmc.samplers
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
