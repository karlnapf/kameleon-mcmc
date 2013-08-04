"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from abc import abstractmethod
from main.kernel.Kernel import Kernel

class Covariance(Kernel):
    def __init__(self):
        Kernel.__init__(self)
        
    @abstractmethod
    def compute(self, X, Y=None):
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