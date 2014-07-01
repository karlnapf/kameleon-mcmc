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


from numpy import zeros, inner, diag, allclose, fill_diagonal
import numpy

from kameleon_mcmc.distribution.Distribution import Distribution
from kameleon_mcmc.tools.GenericTests import GenericTests


class Hopfield(Distribution):
    """
    Defines a distribution on the hypercube given by the Hopfield network 
    (unrestricted Boltzmann machine) with P(x)\propto exp(x'*bias + x'*W*x)
    """
    def __init__(self, W, bias):
        GenericTests.check_type(W, 'W', numpy.ndarray, 2)
        GenericTests.check_type(bias, 'bias', numpy.ndarray, 1)
        
        if not W.shape[0] == W.shape[1]:
            raise ValueError("W must be square")
        
        if not bias.shape[0] == W.shape[0]:
            raise ValueError("dimensions of W and bias must agree")
        
        if not all(diag(W) == 0):
            raise ValueError("W must have zeros along the diagonal")
        
        if not allclose(W, W.T):
            raise ValueError("W must be symmetric")
        
        Distribution.__init__(self, W.shape[0])
        
        self.W = W
        self.bias = bias
    
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
        GenericTests.check_type(X, 'X', numpy.ndarray, 2)
        # this also enforces correct data ranges
        if X.dtype != numpy.bool8:
            raise ValueError("X must be a bool8 numpy array")
            
        if not X.shape[1] == self.dimension:
            raise ValueError("Dimension of X does not match own dimension")
            
        result = zeros(len(X))
        for i in range(len(X)):
            result[i] = inner(X[i], self.bias + self.W.dot(X[i]))
        return result
    
    @staticmethod
    def weights_from_patterns(P):
        """
        Computes a weight matrix so that the network has stationary points at a
        number of given patterns,
        
        W_{ij} = \sum_{i=1}^n (2 p_i^n - 1)(2 p_j^n - 1) and
        W_{ii} = 0
        
        where p_i^n is the i-th bit of the n-th pattern vector given in P.
        (Note the conversion from {0,1} to {-1,+1} via 2x-1.)) 
        
        (Bias should be set to zero.)
        """
        GenericTests.check_type(P, "P", numpy.ndarray, required_shapelen=2)
        
        dim = P.shape[1]
        n = P.shape[0]
        
        if n <= 0:
            raise ValueError("Need at least one pattern.")

        # train network
        W = zeros((dim, dim))
        for i in range(dim):
            for j in range(i, dim):
                for mu in range(n):
                    W[i, j] += (2 * P[mu][i] - 1) * (2 * P[mu][j] - 1)
                
                # W[i,j] /= n
                W[j, i] = W[i, j]
        
        fill_diagonal(W, 0)
        return W
        
