"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from abc import abstractmethod
from numpy import eye
from numpy.ma.core import shape, mean, sqrt
from numpy.lib.index_tricks import fill_diagonal

class Kernel(object):
    def __init__(self):
        pass
    
    def __str__(self):
        s=self.__class__.__name__+ "=[]"
        return s
    
    @abstractmethod
    def kernel(self, X, Y=None):
        raise NotImplementedError()
    
    @abstractmethod
    def gradient(self, x, Y):
        
        # ensure this in every implementation
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        raise NotImplementedError()
    
    @staticmethod
    def centring_matrix(n):
        """
        Returns the centering matrix eye(n) - 1.0 / n
        """
        return eye(n) - 1.0 / n
    
    @staticmethod
    def center_kernel_matrix(K):
        """
        Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
        """
        n = shape(K)[0]
        H = eye(n) - 1.0 / n
        return  1.0 / n * H.dot(K.dot(H))
    
    @abstractmethod
    def estimateMMD(self,sample1,sample2,unbiased=False):
        """
        Compute the MMD between two samples
        """
        K11 = self.kernel(sample1,sample1)
        K22 = self.kernel(sample2,sample2)
        K12 = self.kernel(sample1,sample2)
        if unbiased:
            fill_diagonal(K11,0.0)
            fill_diagonal(K22,0.0)
            n=float(shape(K11)[0])
            m=float(shape(K22)[0])
            return sqrt( sum(sum(K11))/(pow(n,2)-n) + sum(sum(K22))/(pow(m,2)-m) - 2*mean(K12[:]) )
        else:
            return sqrt( mean(K11[:])+mean(K22[:])-2*mean(K12[:]) )