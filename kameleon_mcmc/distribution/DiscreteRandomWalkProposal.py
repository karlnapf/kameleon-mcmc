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

from numpy import arange, where
import numpy
from numpy.matlib import repmat
from numpy.random import rand, randint, permutation

from kameleon_mcmc.distribution.Distribution import Distribution, Sample


class DiscreteRandomWalkProposal(Distribution):
    def __init__(self, mu, spread):
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
        
        self.mu = mu
        self.spread = spread
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "spread=" + str(self.ps)
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
        
        for n in range(n):
            # sample Bernoulli to get number of changes
            num_changes = sum(rand(self.dimension) < self.spread)
            
            # sample action (add=0,del=1,swap=2), has some advantages to represent
            # like this, see below
            action = randint(0, 3)
            
            # check that adding/deleting the desired number is possible,
            # truncate otherwise
            num_pos = sum(self.mu)
            if action is 0 and (num_changes + num_pos) > len(self.mu):
                num_changes = len(self.mu) - num_pos
            elif action is 1 and (num_changes > num_pos):
                num_changes = num_pos
            elif action is 2 and num_changes > min(num_pos, self.dimension - num_pos):
                num_changes = min(num_pos, self.dimension - num_pos)
                
            # if no changes, directly return
            if num_changes == 0:
                return Sample(samples)
            
            # do action, since we chacked the number of changes above, no checks here
            if action is 0 or action is 1:
                # do action: add/delete
                # add or delete is almost the same, below changes only the value in
                # the mean vector to look for
                value_to_change = (action == 1)
                relevant_indices = where(self.mu == value_to_change)[0]
                selected = permutation(arange(len(relevant_indices)))[:num_changes]
                changes = relevant_indices[selected]
                samples[n][changes] = 1
            elif action is 2:
                # do action: swap
                pos_indices = where(self.mu == 1)
                neg_indices = where(self.mu == 0)
                selected_pos = permutation(arange(len(pos_indices)))[:num_changes]
                selected_neg = permutation(arange(len(neg_indices)))[:num_changes]
                changes_pos = pos_indices[selected_pos]
                changes_neg = neg_indices[selected_neg]
                samples[n][changes_pos] = 0
                samples[n][changes_neg] = 1
            
        return Sample(samples)
    
    def log_pdf(self, X):
        if not type(X) is numpy.ndarray:
            raise TypeError("X must be a numpy array")
        
        if not len(X.shape) is 2:
            raise TypeError("X must be a 2D numpy array")
        
        # this also enforce correct data ranges
        if X.dtype != numpy.bool8:
            raise ValueError("X must be a bool8 numpy array")
        
        if not X.shape[1] == self.dimension:
            raise ValueError("Dimension of X does not match own dimension")

        return rand(len(X))

