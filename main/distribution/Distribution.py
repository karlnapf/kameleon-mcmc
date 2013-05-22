from abc import abstractmethod, ABCMeta
from numpy.ma.core import ceil, arange, shape
class Distribution(object):
    __metaclass__ = ABCMeta
     
    def __init__(self, dimension):
        self.dimension = dimension
    
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
        Z = self.sample(1000)
        return zip(ceil(Z.min(0)), ceil(Z.max(0)))
    
