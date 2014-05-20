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

from numpy import logical_xor, sum
import numpy
from numpy.random import randn

from kameleon_mcmc.distribution.Distribution import Distribution
from kameleon_mcmc.distribution.proposals.DiscreteRandomWalkProposal import DiscreteRandomWalkProposal
from kameleon_mcmc.kernel.Kernel import Kernel
from kameleon_mcmc.mcmc.samplers.MCMCSampler import MCMCSampler


class DiscreteKameleon(MCMCSampler):
    """
    Kameleon MCMC on discrete domains, non-adaptive version that takes a set of
    oracle samples.
    """
    def __init__(self, distribution, kernel, Z, threshold, spread):
        if not isinstance(distribution, Distribution):
            raise TypeError("Target must be a Distribution object")
        
        if not isinstance(kernel, Kernel):
            raise TypeError("Kernel must be a Kernel object")
        
        if not type(Z) is numpy.ndarray:
            raise TypeError("History must be a numpy array")
        
        if not len(Z.shape) == 2:
            raise ValueError("History must be a 2D numpy array")
        
        if not Z.shape[1] == distribution.dimension:
            raise ValueError("History dimension does not match target dimension")
        
        if not Z.shape[0] > 0:
            raise ValueError("History must contain at least one point")
        
        if not type(threshold) is float:
            raise TypeError("Threshold must be a float")
        
        if not type(spread) is float:
            raise TypeError("Spread must be a float")
        
        if not (spread > 0. and spread < 1.):
            raise ValueError("Spread must be a probability")
        
        MCMCSampler.__init__(self, distribution)
        
        self.kernel = kernel
        self.Z = Z
        self.threshold = threshold
        self.spread = spread
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "kernel=" + str(self.kernel)
        s += ", threshold=" + str(self.threshold)
        s += ", spread=" + str(self.spread)
        s += ", " + MCMCSampler.__str__(self)
        s += "]"
        return s
    
    def construct_proposal(self, y):
        k = self.kernel.kernel(y.reshape(1,len(y)), self.Z)
        
        # take care about bool8 overflows preventing larger values
        diff = y.astype(numpy.int64)
        diff = diff - self.Z
        beta = randn(len(self.Z))
        weighted_sum = sum((k * beta).T * diff, 0)
        thresholded_sum = weighted_sum > self.threshold
        xored = logical_xor(thresholded_sum, y)
        
        # return distribution object that adds noise to the xor point
        return DiscreteRandomWalkProposal(xored, self.spread)
    
    def adapt(self, mcmc_chain, step_output):
        """
        Nothing for this one since it uses oracle samples
        """
        pass
    
