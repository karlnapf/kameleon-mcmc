from abc import abstractmethod, ABCMeta
from numpy.ma.core import ceil, arange
class Distribution(object):
    __metaclass__ = ABCMeta
     
    def __init__(self, dimension):
        self.dimension = dimension
    
    @abstractmethod
    def sample(self, n=1):
        raise NotImplementedError()
    
    @abstractmethod
    def log_pdf(self, X):
        raise NotImplementedError()
    
    def emp_quantiles(self,X,quantiles=arange(0.9,0,-0.1)):
        raise NotImplementedError()
    
    def get_plotting_bounds(self):
        """
        Samples 1000 points and returns a list of tuples with minimum and
        maximum for each dimension
        """
        Z=self.sample(1000)
        return zip(ceil(Z.min(0)), ceil(Z.max(0)))
    
