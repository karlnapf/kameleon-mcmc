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

from numpy import hstack, arange, sqrt, asarray
from numpy.linalg.linalg import cholesky
from numpy.random import randn

from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.distribution.full_conditionals.FullConditionals import FullConditionals
from kameleon_mcmc.tools.MatrixTools import MatrixTools


class GaussianFullConditionals(FullConditionals):
    """
    Implements the full conditional distributions for a multivariate Gaussian.
    Takes an instance of a Gaussian and computes the full conditionals.
    """
    def __init__(self, full_target, current_state, schedule="in_turns", index_block=None):
        if not isinstance(full_target, Gaussian):
            raise TypeError("Given full Gaussian is not a Gaussian")
        
        FullConditionals.__init__(self, full_target, current_state, schedule, index_block)
        
        # precompute full covariance matrix for slicing later
        self.full_Sigma = full_target.L.dot(full_target.L.T)
        
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "full_Sigma=" + str(self.full_Sigma)
        s += ", " + FullConditionals.__str__(self)
        s += "]"
        return s
    
    def sample_conditional(self, index):
        if index < 0 or index >= self.dimension:
            raise ValueError("Conditional index out of bounds")
        
        # all indices but the current
        cond_inds = hstack((arange(0, index), arange(index + 1, self.dimension)))
#         print "conditioning on index %d" % index
#         print "other indices:", cond_inds
        
        # partition the Gaussian x|y, precompute matrix inversion
        mu_x = self.full_target.mu[index]
        Sigma_xx = self.full_Sigma[index, index]
        mu_y = self.full_target.mu[cond_inds]
        Sigma_yy = self.full_Sigma[cond_inds, cond_inds].reshape(len(cond_inds), len(cond_inds))
        L_yy = cholesky(Sigma_yy)
        Sigma_xy = self.full_Sigma[index, cond_inds]
        Sigma_yx = self.full_Sigma[cond_inds, index]
        
        y = self.current_state[cond_inds]
        
        # mu=mu_x+Sigma_xy Sigma_yy^(-1)(y-mu_y)
        mu = mu_x + Sigma_xy.dot(MatrixTools.cholesky_solve(L_yy, y - mu_y))
        
        # solve Sigma=Sigma_xx-Sigma_yy^-1 Sigma_yx=Sigma_xy-Sigma_xy L_yy^(-T)_yy^(-1) Sigma_yx
        Sigma = Sigma_xx - Sigma_xy.dot(MatrixTools.cholesky_solve(L_yy, Sigma_yx))
        
        # return sample from x|y
        conditional_sample = randn() * sqrt(Sigma) + mu
        return conditional_sample
    
    def get_current_state_array(self):
        # this means that this gaussian conditional can only work on single
        # index conditionals, but not on blocks, which is fine for testing
        return asarray(self.current_state).reshape(1, self.dimension)
