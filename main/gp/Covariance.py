"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from abc import abstractmethod
from main.kernel.Kernel import Kernel

class Covariance(Kernel):
    def __init__(self):
        Kernel.__init__(self)
        
    @abstractmethod
    def get_num_parameters(self):
        raise NotImplementedError()
    
    @abstractmethod
    def set_theta(self, theta):
        raise NotImplementedError()
    
    @abstractmethod
    def get_theta(self):
        raise NotImplementedError()
    
    @abstractmethod
    def compute(self, X, Y=None):
        raise NotImplementedError()