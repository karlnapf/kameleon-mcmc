"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from abc import abstractmethod, ABCMeta
from numpy import ceil, arange, shape

class Distribution(object):
    __metaclass__ = ABCMeta
     
    def __init__(self, dimension):
        self.dimension = dimension
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "dimension="+ str(self.dimension)
        s += "]"
        return s
    
    def sample(self, n=1):
        raise NotImplementedError()
    
    @abstractmethod
    def log_pdf(self, X):
        """
        parameters:
        X - 2D array of row vectors to compute the log-pdf of
        
        returns:
        1D array of log-pdfs of all inputs
        """
        
        # ensure this in every implementation
        assert(len(shape(X))==2)
        assert(shape(X)[1]==self.dimension)
        
        raise NotImplementedError()
    
    def emp_quantiles(self, X, quantiles=arange(0.1, 1, 0.1)):
        raise NotImplementedError()
    
    def get_plotting_bounds(self):
        """
        Samples 1000 points and returns a list of tuples with minimum and
        maximum for each dimension
        """
        Z = self.sample(1000).samples
        return zip(ceil(Z.min(0)), ceil(Z.max(0)))
    
    def get_proposal_points(self, n):
        """
        Returns n points for proposal distributions which might be interesting
        """
        return self.sample(n).samples

class Sample(object):
    def __init__(self,samples):
        self.samples=samples