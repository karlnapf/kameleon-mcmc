"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from kameleon_mcmc.kernel.Kernel import Kernel

class LinearKernel(Kernel):
    def __init__(self):
        Kernel.__init__(self)
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "" + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y):
        """
        Computes the linear kernel k(x,y)=x^T y for the given data
        X - samples on right hand side
        Y - samples on left hand side, can be None in which case its replaced by X
        """
        if Y is None:
            Y = X
        
        return X.dot(Y.T)

    def gradient(self, x, Y, args_euqal=False):
        """
        Computes the linear kernel k(x,y)=x^T y for the given data
        x - single sample on right hand side
        Y - samples on left hand side
        """
        return Y
