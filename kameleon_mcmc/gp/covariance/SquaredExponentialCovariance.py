"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from abc import abstractmethod
from kameleon_mcmc.gp.covariance.Covariance import Covariance
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
from numpy.ma.core import array, shape

class SquaredExponentialCovariance(Covariance):
    def __init__(self, sigma, scale):
        """
        sigma - kernel width 
        scale - kernel scalling
        """
        Covariance.__init__(self)
        self.kernel=GaussianKernel(sigma);
        self.scale=scale
    
    def compute(self, X, Y=None):
        return (self.scale**2)*self.kernel.kernel(X, Y)
    
    @abstractmethod
    def gen_num_hyperparameters(self):
        return 2
    
    @abstractmethod
    def get_hyperparameters(self):
        return array([self.sigma, self.scale])
    
    @abstractmethod
    def set_hyperparameters(self, theta):
        assert(len(shape(theta))==1)
        assert(len(theta)==2)
        
        self.kernel.sigma=theta[0]
        self.kernel.scale=theta[1]
        