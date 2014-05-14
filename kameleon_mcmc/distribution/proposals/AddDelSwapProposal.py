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

from numpy import arange, where, zeros, log, sum, inf
import numpy
from numpy.matlib import repmat
from numpy.random import rand, randint, permutation

from kameleon_mcmc.distribution.Distribution import Distribution, Sample
from kameleon_mcmc.tools.HelperFunctions import HelperFunctions
from kameleon_mcmc.tools.GenericTests import GenericTests


class AddDelSwapProposal(Distribution):
    def __init__(self, mu, spread, N=3):
        GenericTests.check_type(mu, 'mu', numpy.ndarray, 1)
        GenericTests.check_type(spread, 'spread', float)
        GenericTests.check_type(N, 'N', int)
        
        if mu.dtype != numpy.bool8:
            raise ValueError("Mean must be a bool8 numpy array")
        
        Distribution.__init__(self, len(mu))
        
        
        if not (spread > 0. and spread < 1.):
            raise ValueError("Spread must be a probability")
        
        self.mu = mu
        self.spread = spread
        self.N = N
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "mu=" + str(self.mu)
        s += "spread=" + str(self.spread)
        s += "N=" + str(self.N)
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
        
        for i in range(n):
            num_active = sum(self.mu)

            # sample Bernoulli to get number of changes
            # N-1 Bernoulli trials, then add one afterwards
            
            num_changes = sum(rand(self.N-1) < self.spread) + 1
            
            # sample action (add=0,del=1,swap=2), has some advantages to represent
            # like this, see below
            # if less active states than changes, always add
            if num_changes > num_active:
                action = 0
            # if less non-active states than changes, always delete
            elif num_changes > self.dimension - num_active:
                action = 1
            else:
                action = randint(0, 3)
            #print 'action:'
            #print action
            #print 'num_changes:'
            #print num_changes
            # check that adding/deleting the desired number is possible,
            # truncate otherwise, this is needed because if all elements are active
            # then adding is not possible, but we always have at least one change
            # this is a hack and we should email the authors about this
            #if action is 0 and (num_changes + num_active) > self.dimension:
            #    num_changes = 0
            #elif action is 1 and (num_changes > num_active):
            #    num_changes = 0
            #elif action is 2 and num_changes > max_possible_change:
            #    num_changes = 0
            #
            # if no changes, directly return
            #if num_changes == 0:
            #    return Sample(samples)
            # do action, since we chacked the number of changes above, no checks here
            if action is 0 or action is 1:
                # do action: add/delete
                # add or delete is almost the same, below changes only the value in
                # the mean vector to look for
                relevant_indices = where(self.mu == action)[0]
                selected = permutation(arange(len(relevant_indices)))[:num_changes]
                changes = relevant_indices[selected]
                samples[i][changes] = (action == 0)
            elif action is 2:
                # do action: swap
                pos_indices = where(self.mu == 1)[0]
                neg_indices = where(self.mu == 0)[0]
                selected_pos = permutation(arange(len(pos_indices)))[:num_changes]
                selected_neg = permutation(arange(len(neg_indices)))[:num_changes]
                changes_pos = pos_indices[selected_pos]
                changes_neg = neg_indices[selected_neg]
                samples[i][changes_pos] = 0
                samples[i][changes_neg] = 1
            #print samples[i]
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

        num_active_self = sum(self.mu)
        #max_possible_change = min(num_active_self, self.dimension - num_active_self)
        
        # result vector
        log_liks = zeros(len(X))
        
        # compute action dependent log likelihood parts
        for i in range(len(X)):
            x = X[i]
            
            num_active_x = sum(x)
            
            # hamming distances using numpy broadcasting
            # divide by two, integer division is always fine since even number of differences
            num_diff = sum(self.mu != x)
            if num_active_self == num_active_x:
                num_diff / 2
                
            if num_diff > self.N:
                log_liks[i]=-inf
                continue
                
            if num_active_self != num_active_x:
                action = num_active_x < num_active_self
                if not all(x[self.mu==action]==action):
                    log_liks[i]=-inf
                    continue
            else:
                action = 2
            
            #shared-terms
            log_liks[i] = HelperFunctions.log_bin_coeff(self.N - 1, num_diff - 1) \
                            + (num_diff - 1) * log(self.spread) \
                            + (self.N - num_diff) * log(1 - self.spread)
            # if there was a freedom of action, use factor 1/3
            if num_diff <= min(num_active_self,self.dimension-num_active_self):
                log_liks[i] -= log(3)
            # action-specific terms
            if action == 0:
                # add
                log_liks[i] -= HelperFunctions.log_bin_coeff(self.dimension - num_active_self, num_diff)
            elif action == 1:
                # del
                log_liks[i] -= HelperFunctions.log_bin_coeff(num_active_self, num_diff)
            elif action == 2:
                # swap
                log_liks[i] -= HelperFunctions.log_bin_coeff(num_active_self, num_diff) \
                                 - HelperFunctions.log_bin_coeff(self.dimension - num_active_self, num_diff)
            
        return log_liks

