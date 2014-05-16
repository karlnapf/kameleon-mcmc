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
from numpy import zeros, eye, pi, concatenate, array

from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.distribution.full_conditionals.GaussianFullConditionals import GaussianFullConditionals
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.samplers.Gibbs import Gibbs
from kameleon_mcmc.tools.MatrixTools import MatrixTools
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
import time


def main():
    # covariance has stretched Eigenvalues, and rotated basis
    Sigma1 = eye(2)
    Sigma1[0, 0] = 30.0
    Sigma1[1, 1] = 1.0
    Sigma2 = Sigma1
    Sigma2[0, 0] = 20.0
    theta = -pi / 4
    U = MatrixTools.rotation_matrix(theta)
    Sigma1 = U.T.dot(Sigma1).dot(U)
    Sigma2 = U.T.dot(Sigma2).dot(U)
    
    gaussian1 = Gaussian(Sigma=Sigma1)
    gaussian2 = Gaussian(mu=array([1, 0]), Sigma=Sigma1)
    
    oracle_samples1 = gaussian1.sample(n=200).samples
    oracle_samples2 = gaussian2.sample(n=200).samples
    
    distribution1 = GaussianFullConditionals(gaussian1, [0., 0.])
    distribution2 = GaussianFullConditionals(gaussian2, [1., 0.])
    
    mcmc_sampler1 = Gibbs(distribution1)
    mcmc_sampler2 = Gibbs(distribution2)
    
    start = zeros(2)
    mcmc_params = MCMCParams(start=start, num_iterations=2200, burnin=2000)
    
    chain1 = MCMCChain(mcmc_sampler1, mcmc_params)
    chain1.run()
    gibbs_samples1 = chain1.get_samples_after_burnin()
    
    chain2 = MCMCChain(mcmc_sampler2, mcmc_params)
    chain2.run()
    gibbs_samples2 = chain2.get_samples_after_burnin()
    
    sigma = GaussianKernel.get_sigma_median_heuristic(concatenate((oracle_samples1,oracle_samples2),axis=0))
    kernel = GaussianKernel(sigma=sigma)
    
    print '...running the oracle1<->oracle2 tests'
    print 'mmd (oracle1<->oracle2): ', kernel.estimateMMD(oracle_samples1,oracle_samples2)
    
    start=time.time()
    print 'p-value (vanilla): ', kernel.TwoSampleTest(oracle_samples1,oracle_samples2,method='vanilla')
    end=time.time()
    print 'time elapsed:', end-start
    
    start=time.time()
    print 'p-value (block): ', kernel.TwoSampleTest(oracle_samples1,oracle_samples2,method='block')
    end=time.time()
    print 'time elapsed:', end-start
    
    start=time.time()
    print 'p-value (wild): ', kernel.TwoSampleTest(oracle_samples1,oracle_samples2,method='wild')
    end=time.time()
    print 'time elapsed:', end-start
    
    print '...running the oracle1<->gibbs1 tests'
    print 'mmd (oracle1<->gibbs1): ', kernel.estimateMMD(oracle_samples1,gibbs_samples1)
    
    start=time.time()
    print 'p-value (vanilla): ', kernel.TwoSampleTest(oracle_samples1,gibbs_samples1,method='vanilla')
    end=time.time()
    print 'time elapsed:', end-start
    
    start=time.time()
    print 'p-value (block): ', kernel.TwoSampleTest(oracle_samples1,gibbs_samples1,method='block')
    end=time.time()
    print 'time elapsed:', end-start
    
    start=time.time()
    print 'p-value (wild): ', kernel.TwoSampleTest(oracle_samples1,gibbs_samples1,method='wild')
    end=time.time()
    print 'time elapsed:', end-start
    
    print '...running the oracle2<->gibbs1 tests'
    print 'mmd (oracle2<->gibbs1): ', kernel.estimateMMD(oracle_samples2,gibbs_samples1)
    
    start=time.time()
    print 'p-value (vanilla): ', kernel.TwoSampleTest(oracle_samples2,gibbs_samples1,method='vanilla')
    end=time.time()
    print 'time elapsed:', end-start
    
    start=time.time()
    print 'p-value (block): ', kernel.TwoSampleTest(oracle_samples2,gibbs_samples1,method='block')
    end=time.time()
    print 'time elapsed:', end-start
    
    start=time.time()
    print 'p-value (wild): ', kernel.TwoSampleTest(oracle_samples2,gibbs_samples1,method='wild')
    end=time.time()
    print 'time elapsed:', end-start

main()
