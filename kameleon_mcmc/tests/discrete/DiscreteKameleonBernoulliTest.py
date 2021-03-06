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

from numpy import zeros, mean
import numpy
from numpy.linalg import norm
from numpy.random import rand

from kameleon_mcmc.distribution.Bernoulli import Bernoulli
from kameleon_mcmc.kernel.HypercubeKernel import HypercubeKernel
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.DiscretePlottingOutput import DiscretePlottingOutput
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.DiscreteKameleon import DiscreteKameleon


def main():
    d = 5
    ps = rand(d)
    ps /= norm(ps)
    distribution = Bernoulli(ps)
    
    num_history = 100
    Z = distribution.sample(num_history).samples
    threshold = 0.8
    spread = 0.2
    
    gamma = 0.2
    kernel = HypercubeKernel(gamma)
    
    mcmc_sampler = DiscreteKameleon(distribution, kernel, Z, threshold, spread)
    
    start = zeros(distribution.dimension, dtype=numpy.bool8)
    mcmc_params = MCMCParams(start=start, num_iterations=1000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(StatisticsOutput(plot_times=True))
    chain.append_mcmc_output(DiscretePlottingOutput(plot_from=0, lag=100))
    
    chain.run()
    print "ps", ps
    print "empirical", mean(chain.samples, 0)
main()
