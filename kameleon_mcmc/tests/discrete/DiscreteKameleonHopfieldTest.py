"""
Copyright (c) 2013-2014 Heiko Strathmann, Dino Sejdinovic
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 *
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 *
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the author.
"""

from numpy import zeros, fill_diagonal, asarray
import numpy
from numpy.random import rand, randn, permutation

from kameleon_mcmc.distribution.Hopfield import Hopfield
from kameleon_mcmc.distribution.full_conditionals.HopfieldFullConditionals import HopfieldFullConditionals
from kameleon_mcmc.kernel.HypercubeKernel import HypercubeKernel
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.DiscretePlottingOutput import DiscretePlottingOutput
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.DiscreteKameleon import DiscreteKameleon
from kameleon_mcmc.mcmc.samplers.Gibbs import Gibbs


def sample_gibbs(distribution, num_samples=1000):
    mcmc_sampler = Gibbs(distribution)
    current_state = [rand() < 0.5 for _ in range(distribution.dimension)]
    mcmc_params = MCMCParams(start=asarray(current_state, dtype=numpy.bool8), num_iterations=num_samples)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    chain.run()
    return chain.samples

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
    
    print("Running Gibbs to produce chain history")
    Z = sample_gibbs(distribution, num_samples=2000)[1500:].astype(numpy.bool8)
    inds = permutation(len(Z))
    Z = Z[inds[:500]]
    print("done")
    
    threshold = 0.8
    spread = 0.2
    
    gamma = 0.2
    kernel = HypercubeKernel(gamma)
    
    mcmc_sampler = DiscreteKameleon(hopfield, kernel, Z, threshold, spread)
    
    start = zeros(distribution.dimension, dtype=numpy.bool8)
    mcmc_params = MCMCParams(start=start, num_iterations=10000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(StatisticsOutput(plot_times=True))
    chain.append_mcmc_output(DiscretePlottingOutput(plot_from=0, lag=500))
    
    chain.run()

main()
