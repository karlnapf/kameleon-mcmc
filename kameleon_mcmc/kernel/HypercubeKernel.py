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
from numpy import tanh
import numpy
from scipy.spatial.distance import squareform, pdist, cdist

from kameleon_mcmc.kernel.Kernel import Kernel


class HypercubeKernel(Kernel):
    def __init__(self, gamma):
        Kernel.__init__(self)
        
        if type(gamma) is not float:
            raise TypeError("Gamma must be float")
        
        self.gamma = gamma
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "gamma=" + str(self.gamma)
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        """
        Computes the hypercube kernel k(x,y)=tanh(gamma)^d(x,y), where d is the
        Hamming distance between x and y
        
        X - 2d numpy.bool8 array, samples on right left side
        Y - 2d numpy.bool8 array, samples on left hand side.
            Can be None in which case its replaced by X
        """
        
        if not type(X) is numpy.ndarray:
            raise TypeError("X must be numpy array")
        
        if not len(X.shape) == 2:
            raise ValueError("X must be 2D numpy array")
        
        if not X.dtype == numpy.bool8:
            raise ValueError("X must be boolean numpy array")
        
        if not Y is None:
            if not type(Y) is numpy.ndarray:
                raise TypeError("Y must be None or numpy array")
            
            if not len(Y.shape) == 2:
                raise ValueError("Y must be None or 2D numpy array")
            
            if not Y.dtype == numpy.bool8:
                raise ValueError("Y must be boolean numpy array")
        
            if not X.shape[1] == Y.shape[1]:
                raise ValueError("X and Y must have same dimension if Y is not None")
        
        # un-normalise normalised hamming distance in both cases
        if Y is None:
            K = tanh(self.gamma) ** squareform(pdist(X, 'hamming') * X.shape[1])
        else:
            K = tanh(self.gamma) ** (cdist(X, Y, 'hamming') * X.shape[1])
            
        return K
    
    def gradient(self, x, Y):
        """
        Computes the gradient of the hypercube kernel wrt. to the left argument
        
        x - single sample on right hand side (1D vector)
        Y - samples on left hand side (2D matrix)
        """
        pass

