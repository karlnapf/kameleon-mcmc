from main.distribution.Distribution import Distribution, Sample
from main.distribution.Gaussian import Gaussian
from numpy.core.function_base import linspace
from numpy.core.shape_base import hstack
from numpy.lib.twodim_base import eye
from numpy.ma.core import sqrt, arange, zeros, shape, array
from numpy.random import randn

class Banana(Distribution):
    '''
    Banana distribution from Haario et al, 1999
    '''
    def __init__(self, dimension=2, bananicity=0.03, V=100.0):
        assert(dimension >= 2)
        Distribution.__init__(self, dimension)
        
        self.bananicity = bananicity
        self.V = V
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "bananicity=" + str(self.bananicity)
        s += ", V=" + str(self.V)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s
    
    def sample(self, n=1):
        X = randn(n, 2)
        X[:, 0] = sqrt(self.V) * X[:, 0]
        X[:, 1] = X[:, 1] + self.bananicity * (X[:, 0] ** 2 - self.V)
        if self.dimension > 2:
            X = hstack((X, randn(n, self.dimension - 2)))
            
        return Sample(X)
    
    def log_pdf(self, X):
        assert(len(shape(X)) == 2)
        assert(shape(X)[1] == self.dimension)
        
        transformed = X.copy()
        transformed[:, 1] = X[:, 1] - self.bananicity * ((X[:, 0] ** 2) - self.V)
        transformed[:, 0] = X[:, 0] / sqrt(self.V)
        phi = Gaussian(zeros([self.dimension]), eye(self.dimension))
        return phi.log_pdf(transformed)
    
    def emp_quantiles(self, X, quantiles=arange(0.1, 1, 0.1)):
        assert(len(shape(X)) == 2)
        assert(shape(X)[1] == self.dimension)
        
        transformed = X.copy()
        transformed[:, 1] = X[:, 1] - self.bananicity * ((X[:, 0] ** 2) - self.V)
        transformed[:, 0] = X[:, 0] / sqrt(self.V)
        phi = Gaussian(zeros([self.dimension]), eye(self.dimension))
        return phi.emp_quantiles(transformed, quantiles)
    
    def get_plotting_bounds(self):
        if self.bananicity == 0.03 and self.V == 100.0:
            return [(-20, 20), (-7, 12)]
        elif self.bananicity == 0.1 and self.V == 100.0:
            return [(-20, 20), (-5, 30)]
        else:
            return Distribution.get_plotting_bounds(self)

    def get_proposal_points(self, n):
        """
        Returns n points which lie on a uniform grid on the "center" of the banana
        """
        if self.dimension == 2:
            (xmin, xmax), _ = self.get_plotting_bounds()
            x1 = linspace(xmin, xmax, n)
            x2 = self.bananicity * (x1 ** 2 - self.V)
            return array([x1, x2]).T
        else:
            return Distribution.get_proposal_points(self, n)
