"""
Copyright (c) 2013-2014 Heiko Strathmann
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

from numpy import mod, log, sum, zeros, arange, inf
import numpy
from numpy.matlib import repmat
from numpy.random import rand, permutation

from kameleon_mcmc.distribution.Distribution import Distribution, Sample


class DiscreteRandomWalkProposal(Distribution):
    def __init__(self, mu, spread, flip_at_least_one=True):
        if not type(mu) is numpy.ndarray:
            raise TypeError("Mean vector must be a numpy array")
        
        if not len(mu.shape) == 1:
            raise ValueError("Mean vector must be a 1D numpy array")
        
        if not len(mu) > 0:
            raise ValueError("Mean vector dimension must be positive")
        
        if mu.dtype != numpy.bool8:
            raise ValueError("Mean must be a bool8 numpy array")
        
        Distribution.__init__(self, len(mu))

        if not type(spread) is float:
            raise TypeError("Spread must be a float")
        
        if not (spread > 0. and spread < 1.):
            raise ValueError("Spread must be a probability")
        
        if not type(flip_at_least_one) is bool:
            raise ValueError("Flip at least one must be a boolean")
        
        self.mu = mu
        self.spread = spread
        self.flip_at_least_one = flip_at_least_one
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "spread=" + str(self.ps)
        s += ", flip_at_least_one=" + str(self.flip_at_least_one)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s
    
    def sample(self, n=1):
        if not type(n) is int:
            raise TypeError("Number of samples must be integer")

        if n <= 0:
            raise ValueError("Number of samples (%d) needs to be positive", n)
        
        # copy mean vector a couple of times
        samples = repmat(self.mu, n, 1)
        
        if self.flip_at_least_one is False:
            # indices to flip, evenly distributed and the change probability is Bernoulli
            change_inds = rand(n, self.dimension) < self.spread
        else:
            # sample number of changes from binomial(spread, d-1) to have at least one change
            num_changes = 1 + sum(rand(n, self.dimension - 1) < self.spread, 1)
            
            # randomly change that many indices
            change_inds = zeros((n, self.dimension), dtype=numpy.bool8)
            for i in range(n):
                change_inds[i, arange(num_changes[i])] = True
                change_inds[i] = change_inds[i, permutation(self.dimension)]
        
        # flip all chosen indices
        samples[change_inds] = mod(samples[change_inds] + 1, 2)
        
        return Sample(samples)
    
    def log_pdf(self, X):
        if not type(X) is numpy.ndarray:
            raise TypeError("X must be a numpy array")
        
        if not len(X.shape) is 2:
            raise TypeError("X must be a 2D numpy array")
        
        # this also enforces correct data ranges
        if X.dtype != numpy.bool8:
            raise ValueError("X must be a bool8 numpy array")
        
        if not X.shape[1] == self.dimension:
            raise ValueError("Dimension of X does not match own dimension")

        # hamming distance for all elements in X
        k = sum(X != self.mu, 1)
        
        # remove from distance as its always flipped
        if self.flip_at_least_one:
            k -= 1
        
        # simple binomial probability for d-1 dimensions, where the normaliser cancel
        
        d = self.dimension
        if self.flip_at_least_one:
            # one index is always flipped, so exclude
            d -= 1
            
        result = k * log(self.spread) + (d - k) * log(1 - self.spread)
        
        if self.flip_at_least_one:
            # cases with k<0 have zero probability since one element is *always*
            # changed, return -inf
            result[k < 0] = -inf

        return result

