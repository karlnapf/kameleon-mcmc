"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from numpy import zeros, fill_diagonal
from numpy.random import rand, randn

from kameleon_mcmc.distribution.Hopfield import Hopfield
from kameleon_mcmc.distribution.full_conditionals.HopfieldFullConditionals import HopfieldFullConditionals
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.DiscretePlottingOutput import DiscretePlottingOutput
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.Gibbs import Gibbs


def main():
    d = 5
    b = randn(d)
    V = randn(d, d)
    W = V + V.T
    fill_diagonal(W, zeros(d))
    hopfield = Hopfield(W, b)
    current_state = [rand() < 0.5 for _ in range(d)]
    distribution = HopfieldFullConditionals(full_target=hopfield,
                                            current_state=current_state)
    
    mcmc_sampler = Gibbs(distribution)
    
    start = zeros(distribution.dimension)
    mcmc_params = MCMCParams(start=start, num_iterations=10000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(StatisticsOutput(plot_times=True, lag=100))
    chain.append_mcmc_output(DiscretePlottingOutput(plot_from=0, lag=100))
    chain.run()
    
main()
