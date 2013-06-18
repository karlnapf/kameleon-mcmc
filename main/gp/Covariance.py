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