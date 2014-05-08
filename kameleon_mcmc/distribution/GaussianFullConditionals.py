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

from numpy import hstack, arange, sqrt
from numpy.linalg.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.stats.distributions import norm

from kameleon_mcmc.distribution.FullConditionals import FullConditionals
from kameleon_mcmc.distribution.Gaussian import Gaussian


class GaussianFullConditionals(FullConditionals):
    """
    Implements the full conditional distributions for a multivariate Gaussian.
    Takes an instance of a Gaussian and computes the full conditionals.
    """
    def __init__(self, full_gaussian, current_state, schedule, index_block=None):
        if not isinstance(full_gaussian, Gaussian):
            raise TypeError("Given full Gaussian is not a Gaussian")
        
        FullConditionals.__init__(self, current_state, schedule, index_block)
        
        self.full_gaussian = full_gaussian
        
        # precompute full covariance matrix for slicing later
        self.full_Sigma = full_gaussian.L.dot(full_gaussian.L)
        
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "full_gaussian=" + str(self.gaussian)
        s += ", " + FullConditionals.__str__(self)
        s += "]"
        return s
    
    def sample_conditional(self, index, current):
        if index < 0 or index >= self.dimension:
            raise ValueError("Conditional index out of bounds")
        
        # all indices but the current
        cond_inds = hstack((arange(0, index), arange(index + 1, self.dimension)))
        
        # partition the Gaussian x|y, precompute matrix inversion
        mu_x = self.full_gaussian.mu[index]
        Sigma_xx = self.full_Sigma[index, index]
        mu_y = self.full_gaussian.mu[cond_inds]
        Sigma_yy = self.full_Sigma[cond_inds, cond_inds]
        L_yy = cholesky(Sigma_yy)
        Sigma_xy = self.full_Sigma[index, cond_inds]
        y = self.current_state[index]
        
        # solve mu=mu_x+Sigma_yy^(-1)(y-mu_y) = mu_x+L_yy^(-T)_yy^(-1)(y-mu_y)
        mu = y - mu_y
        mu = solve_triangular(L_yy, mu.T, lower=True)
        mu = solve_triangular(L_yy.T, mu, lower=False)
        mu += mu_x
        
        # solve Sigma=Sigma_xy-Sigma_yy^-1 Sigma_yx=Sigma_xy-L_yy^(-T)_yy^(-1) Sigma_yx
        Sigma = Sigma_xy
        Sigma = solve_triangular(L_yy, Sigma.T, lower=True)
        Sigma = solve_triangular(L_yy.T, Sigma, lower=False)
        Sigma = Sigma_xx - Sigma
        
        # return sample from x|y
        return norm.rvs(mu, sqrt(Sigma))
