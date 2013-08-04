"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from abc import abstractmethod
from numpy.ma.core import asarray

class Likelihood(object):
    def __init__(self):
        pass
        
    @abstractmethod
    def log_lik_vector(self, y, f):
        raise NotImplementedError()
    
    def log_lik_vector_multiple(self, y, F):
        return asarray([self.log_lik_vector(y, f) for f in F])
    
    @abstractmethod
    def log_lik_grad_vector(self, y, f):
        raise NotImplementedError()

    @abstractmethod
    def log_lik_hessian_vector(self, y, f):
        raise NotImplementedError()
    
    @abstractmethod
    def gen_num_hyperparameters(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_hyperparameters(self):
        raise NotImplementedError()
    
    @abstractmethod
    def set_hyperparameters(self, theta):
        raise NotImplementedError()
