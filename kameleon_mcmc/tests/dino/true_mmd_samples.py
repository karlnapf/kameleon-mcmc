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
from numpy import zeros, mean, eye, pi, concatenate, array, reshape, empty, arange, percentile

from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.distribution.full_conditionals.GaussianFullConditionals import GaussianFullConditionals
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.samplers.Gibbs import Gibbs
from kameleon_mcmc.tools.MatrixTools import MatrixTools
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
import time
from kameleon_mcmc.tools.Visualise import Visualise
from matplotlib.pyplot import show, hist, plot
from pickle import dump, load
import sys
from numpy.ma.core import sort


def read_samples():
    f = open("/home/dino/git/mmdIIDTrueSamples.dat", "r")
    H0_samples = load(f)
    HA_samples = load(f)
    gaussian1 = load(f)
    gaussian2 = load(f)
    f.close()
    print 'P1:', gaussian1.mu, gaussian1.L.dot(gaussian1.L.T)
    print 'P2:', gaussian2.mu, gaussian2.L.dot(gaussian2.L.T)
    hist((H0_samples,HA_samples),50)
    th=percentile(H0_samples,95)
    print 'expected Type II: ', sum(HA_samples<th)/float(len(HA_samples))
    #print H0_samples
    #print HA_samples
    show()
    return None

def main():
    numTrials = 500
    n=200
    Sigma1 = eye(2)
    Sigma1[0, 0] = 30.0
    Sigma1[1, 1] = 1.0
    theta = - pi / 4
    U = MatrixTools.rotation_matrix(theta)
    Sigma1 = U.T.dot(Sigma1).dot(U)
    print Sigma1
    gaussian1 = Gaussian(Sigma=Sigma1)
    gaussian2 = Gaussian(mu=array([1., 0.]), Sigma=Sigma1)
    
    oracle_samples1 = gaussian1.sample(n=n).samples
    oracle_samples2 = gaussian2.sample(n=n).samples
    
    print 'mean1:', mean(oracle_samples1,0)
    print 'mean2:', mean(oracle_samples2,0)
    plot(oracle_samples1[:,0],oracle_samples1[:,1],'b*')
    plot(oracle_samples2[:,0],oracle_samples2[:,1],'r*')
    show()
    distribution1 = GaussianFullConditionals(gaussian1, list(gaussian1.mu))
    distribution2 = GaussianFullConditionals(gaussian2, list(gaussian2.mu))
    
    H0_samples = zeros(numTrials)
    HA_samples = zeros(numTrials)
    mcmc_sampler1 = Gibbs(distribution1)
    mcmc_sampler2 = Gibbs(distribution2)
    burnin = 9000
    thin = 5
    start = zeros(2)
    mcmc_params = MCMCParams(start=start, num_iterations=burnin+thin*n, burnin=burnin)
    sigma = GaussianKernel.get_sigma_median_heuristic(concatenate((oracle_samples1,oracle_samples2),axis=0))
    print 'using bandwidth: ', sigma
    kernel = GaussianKernel(sigma=sigma)
    
    for ii in arange(numTrials):
        start =time.time()
        print 'trial:', ii
        
        oracle_samples1 = gaussian1.sample(n=n).samples
        oracle_samples1a = gaussian1.sample(n=n).samples
        oracle_samples2 = gaussian2.sample(n=n).samples
        
        #         chain1 = MCMCChain(mcmc_sampler1, mcmc_params)
        #         chain1.run()
        #         gibbs_samples1 = chain1.get_samples_after_burnin()
        #         gibbs_samples1 = gibbs_samples1[thin*arange(n)]
        #         
        #         chain1a = MCMCChain(mcmc_sampler1, mcmc_params)
        #         chain1a.run()
        #         gibbs_samples1a = chain1a.get_samples_after_burnin()
        #         gibbs_samples1a = gibbs_samples1a[thin*arange(n)]
        #         
        #         chain2 = MCMCChain(mcmc_sampler2, mcmc_params)
        #         chain2.run()
        #         gibbs_samples2 = chain2.get_samples_after_burnin()
        #         gibbs_samples2 = gibbs_samples2[thin*arange(n)]
        
        
        #         H0_samples[ii]=kernel.estimateMMD(gibbs_samples1,gibbs_samples1a)
        #         HA_samples[ii]=kernel.estimateMMD(gibbs_samples1,gibbs_samples2)
        #         
        H0_samples[ii]=kernel.estimateMMD(oracle_samples1,oracle_samples1a)
        HA_samples[ii]=kernel.estimateMMD(oracle_samples1,oracle_samples2)
        end=time.time()
        print 'time elapsed: ', end-start
        
    f = open("/home/dino/git/mmdIIDTrueSamples.dat", "w")
    dump(H0_samples, f)
    dump(HA_samples, f)
    dump(gaussian1, f)
    dump(gaussian2, f)
    f.close()
    return None
main()
read_samples()
