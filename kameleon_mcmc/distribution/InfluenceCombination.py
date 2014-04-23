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

from numpy import asarray
from numpy import log, exp, inner, logaddexp
import numpy
from numpy.matlib import repmat
from numpy.random import rand

from kameleon_mcmc.distribution.Distribution import Distribution, Sample


class InfluenceCombination(Distribution):
    def __init__(self, W, bias):
        if not type(W) is numpy.ndarray:
            raise TypeError("W must be a numpy array")
        if not len(W.shape) is 2:
            raise TypeError("W must be a 2D numpy array")
        if not type(bias) is numpy.ndarray:
            raise TypeError("bias must be a numpy array")
        if not len(bias.shape) is 1:
            raise TypeError("bias must be a 1D numpy array")
            
        Distribution.__init__(self, W.shape[1])        
           
        self.W = W
        self.bias = bias
        self.num_hidden_units = W.shape[0]
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "W=" + str(self.W)    
        s += "bias=" + str(self.bias)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s
    
    def sample(self, n=1):
        if not type(n) is int:
            raise TypeError("Number of samples must be integer")
        raise NotImplementedError()	
    
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
            
        result = zeros(len(X))
        for i in range(len(X)):
            x_psk = [1-2*xx for xx in X[i]]
            result[i]= sum([logaddexp(0,inner(self.W[j],x_psk)+self.bias[j]) for j in range(self.num_hidden_units)])
        return result

