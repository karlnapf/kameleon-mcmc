"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from numpy import array

from kameleon_mcmc.kernel.Kernel import Kernel


class PolynomialKernel(Kernel):
    def __init__(self, degree,theta=0.2):
        Kernel.__init__(self)
        self.degree = degree
        self.theta = theta
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "degree="+ str(self.degree)
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        """
        Computes the polynomial kernel k(x,y)=(1+theta*<x,y>)^degree for the given data
        X - samples on right hand side
        Y - samples on left hand side, can be None in which case its replaced by X
        """
        if Y is None:
            Y = X
        
        return pow(self.theta+X.dot(Y.T), self.degree)
    
    def gradient(self, x, Y):
        """
        Computes the gradient of the Polynomial kernel wrt. to the left argument, i.e.
        \nabla_x k(x,y)=\nabla_x (1+x^Ty)^d=d(1+x^Ty)^(d-1) y
        
        x - single sample on right hand side (1D vector)
        Y - samples on left hand side (2D matrix)
        """
        assert(len(x.shape)==1)
        assert(len(Y.shape)==2)
        assert(len(x)==Y.shape[1])
        
        return self.degree*pow(1+x.dot(Y.T), self.degree)*Y
