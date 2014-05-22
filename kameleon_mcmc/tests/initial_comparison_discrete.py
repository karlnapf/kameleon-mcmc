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

from cPickle import dump, load
from matplotlib.pyplot import plot, show, legend
from numpy import zeros, fill_diagonal, asarray, arange, sqrt, linspace
import numpy
from numpy.random import rand, randn, randint
import time

from kameleon_mcmc.distribution.Hopfield import Hopfield
from kameleon_mcmc.distribution.full_conditionals.HopfieldFullConditionals import HopfieldFullConditionals
from kameleon_mcmc.kernel.HypercubeKernel import HypercubeKernel
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.samplers.DiscreteKameleon import DiscreteKameleon
from kameleon_mcmc.mcmc.samplers.Gibbs import Gibbs
from kameleon_mcmc.mcmc.samplers.StandardMetropolisDiscrete import StandardMetropolisDiscrete


def create_ground_truth():
    filename_chain = "chain.bin"
    filename_Z = "Z.bin"
    filename_hopfield = "hopfield.bin"
    
    try:
        f = open(filename_Z, "r")
        Z = load(f)
        f.close()
        
        f = open(filename_hopfield, "r")
        hopfield = load(f)
        f.close()
        print("Loaded existing ground truth samples and hopfield netword.")
    except IOError:
        print("No existing ground truth samples. Creating.")
        
        # the network to sample from
        try:
            f = open(filename_hopfield, "r")
            hopfield = load(f)
            f.close()
            d = hopfield.dimension
            print("Loaded hopfield network")
        except IOError:
            d = 50
            b = randn(d)
            V = randn(d, d)
            W = V + V.T
            fill_diagonal(W, zeros(d))
            hopfield = Hopfield(W, b)
        
        # dump hopfield network
        f = open(filename_hopfield, "w")
        dump(hopfield, f)
        f.close()
        
        # iterations
        num_iterations = 10000000
        warm_up = 100000
        thin = 2000
        
        current_state = [rand() < 0.5 for _ in range(d)]
        distribution = HopfieldFullConditionals(full_target=hopfield,
                                                current_state=current_state,
                                                schedule="random_permutation")
        mcmc_sampler = Gibbs(distribution)
#         spread = .0001
#         mcmc_sampler = StandardMetropolisDiscrete(hopfield, spread)
        
        mcmc_params = MCMCParams(start=asarray(current_state, dtype=numpy.bool8), num_iterations=num_iterations)
        chain = MCMCChain(mcmc_sampler, mcmc_params)
        
        chain.append_mcmc_output(StatisticsOutput(plot_times=True, lag=1000))
        # chain.append_mcmc_output(StoreChainOutput(".", lag=100000))
        
    #     chain.append_mcmc_output(DiscretePlottingOutput(plot_from=0, lag=100))
        chain.run()
        
        # dump chain
        try:
            f = open(filename_chain, "w")
            dump(chain, f)
            f.close()
        except IOError:
            print("Could not save MCMC chain")
        
        # warmup and thin
        Z = chain.samples[(warm_up):]
        Z = Z[arange(len(Z), step=thin)]
        Z = Z.astype(numpy.bool8)
        
        # dump ground truth samples
        try:
            f = open(filename_Z, "w")
            dump(Z, f)
            f.close()
        except IOError:
            print("Could not save Z")
    
    return Z, hopfield

def run_kameleon_chain(Z, hopfield, start, num_iterations):
    threshold = 0.8
    spread = 0.03
    gamma = 0.2
    kernel = HypercubeKernel(gamma)
    sampler = DiscreteKameleon(hopfield, kernel, Z, threshold, spread)
    params = MCMCParams(start=start, num_iterations=num_iterations)
    chain = MCMCChain(sampler, params)
    chain.run()
    
    return chain

def run_gibbs_chain(hopfield, start, num_iterations):
    d = hopfield.dimension
    current_state = [x for x in start]
    distribution = HopfieldFullConditionals(full_target=hopfield,
                                            current_state=current_state)
    sampler = Gibbs(distribution)
    params = MCMCParams(start=asarray(current_state, dtype=numpy.bool8), num_iterations=num_iterations * d)
    chain = MCMCChain(sampler, params)
    chain.append_mcmc_output(StatisticsOutput(plot_times=True, lag=1000))
    chain.run()
    
    return chain

def run_sm_chain(hopfield, start, num_iterations):
    current_state = [x for x in start]
    spread = 0.03
    sampler = StandardMetropolisDiscrete(hopfield, spread)
    params = MCMCParams(start=asarray(current_state, dtype=numpy.bool8), num_iterations=num_iterations)
    chain = MCMCChain(sampler, params)
    chain.append_mcmc_output(StatisticsOutput(plot_times=True, lag=1000))
    chain.run()
    
    return chain

def main():
    Z, hopfield = create_ground_truth()
    d = hopfield.dimension
    
    print("Number of ground truth samples: %d" % len(Z))

    num_iterations = 200000
    warm_up = 1000
    thin = 100
    
    start = randint(0, 2, d).astype(numpy.bool8)
    timestring = time.strftime("%Y-%m-%d_%H:%M:%S")

    print("Running SM for %d iterations" % num_iterations)
    sm_chain = run_sm_chain(hopfield, start, num_iterations)
    try:
        fname = "temp_sm_result_" + timestring + ".bin"
        f = open(fname, "w")
        dump(sm_chain, f)
        f.close()
    except IOError:
        print("Could not save this SM chain")

    print("Running Gibbs for %d iterations" % (num_iterations * d))
    gibbs_chain = run_gibbs_chain(hopfield, start, num_iterations)
    try:
        fname = "temp_gibbs_result_" + timestring + ".bin"
        f = open(fname, "w")
        dump(gibbs_chain, f)
        f.close()
    except IOError:
        print("Could not save this Gibbs chain")
    
    print("Running Discrete Kameleon for %d iterations" % num_iterations)
    kameleon_chain = run_kameleon_chain(Z, hopfield, start, num_iterations)
    try:
        fname = "temp_kameleon_result_" + timestring + ".bin"
        f = open(fname, "w")
        dump(kameleon_chain, f)
        f.close()
    except IOError:
        print("Could not save this Kameleon chain")
    
    
    # remove warm up and thin
    print("Removing warm up and thinning")
    S_g = gibbs_chain.samples[warm_up:]
    S_g = S_g[arange(len(S_g), step=thin * d)].astype(numpy.bool8)
    S_k = kameleon_chain.samples[warm_up:]
    S_k = S_k[arange(len(S_k), step=thin)].astype(numpy.bool8)
    S_sm = sm_chain.samples[warm_up:]
    S_sm = S_sm[arange(len(S_sm), step=thin)].astype(numpy.bool8)
    print("Gibbs samples: %d" % len(S_g))
    print("Kameleon samples: %d" % len(S_k))
    print("SM samples: %d" % len(S_sm))
    
    
    print("MMDs:")
    kernel = HypercubeKernel(0.2)
    
    num_evaluations = 10
    inds_g = linspace(0, len(S_g), num_evaluations).astype(numpy.int)
    inds_k = linspace(0, len(S_k), num_evaluations).astype(numpy.int)
    inds_sm = linspace(0, len(S_sm), num_evaluations).astype(numpy.int)
    mmds = zeros((3, num_evaluations - 1))
    for i in arange(num_evaluations - 1):
        mmds[0, i - 1] = sqrt(kernel.estimateMMD(S_g[:inds_g[i + 1]], Z))
        mmds[1, i - 1] = sqrt(kernel.estimateMMD(S_k[:inds_k[i + 1]], Z))
        mmds[2, i - 1] = sqrt(kernel.estimateMMD(S_sm[:inds_sm[i + 1]], Z))
        
    
    print(mmds)
    plot(inds_g[1:], mmds[0, :])
    plot(inds_k[1:], mmds[1, :])
    plot(inds_sm[1:], mmds[2, :])
    legend(["Gibbs", "Kameleon", "SM"])
    show()
    
    
if  __name__ == '__main__':
    main()
