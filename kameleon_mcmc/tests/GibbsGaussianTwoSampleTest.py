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
from numpy import zeros, eye, pi, shape

from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.distribution.full_conditionals.GaussianFullConditionals import GaussianFullConditionals
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.samplers.Gibbs import Gibbs
from kameleon_mcmc.tools.MatrixTools import MatrixTools
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel


def main():
    # covariance has stretched Eigenvalues, and rotated basis
    Sigma = eye(2)
    Sigma[0, 0] = 30
    Sigma[1, 1] = 1
    theta = -pi / 4
    U = MatrixTools.rotation_matrix(theta)
    Sigma = U.T.dot(Sigma).dot(U)
    
    gaussian = Gaussian(Sigma=Sigma)
    oracle_samples = gaussian.sample(n=200).samples
    distribution = GaussianFullConditionals(gaussian, [0., 0.])
    
    mcmc_sampler = Gibbs(distribution)
    
    start = zeros(distribution.dimension)
    mcmc_params = MCMCParams(start=start, num_iterations=1200, burnin=1000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    chain.run()
    samples = chain.get_samples_after_burnin()
    
    sigma = GaussianKernel.get_sigma_median_heuristic(oracle_samples)
    kernel = GaussianKernel(sigma=sigma)
    print 'p-value: ', kernel.TwoSampleTest(oracle_samples,samples)
    
main()
