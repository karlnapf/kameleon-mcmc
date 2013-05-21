from main.distribution.Distribution import Distribution
from main.distribution.Gaussian import Gaussian
from numpy.core.shape_base import hstack
from numpy.ma.core import shape, sqrt, arange, zeros
from numpy.random import randn
from numpy.lib.twodim_base import eye

class Banana(Distribution):
    '''
    Banana distribution from Haario et al, 1999
    '''
    def __init__(self, dimension=2, bananicity=0.03, V=100.0):
        assert(dimension >= 2)
        Distribution.__init__(self, dimension)
        
        self.bananicity = bananicity
        self.V = V
        
    def sample(self, n=1):
        X = randn(n, 2)
        X[:, 0] = sqrt(self.V) * X[:, 0]
        X[:, 1] = X[:, 1] + self.bananicity * (X[:, 0] ** 2 - self.V)
        if self.dimension > 2:
            X = hstack((X, randn(n, self.dimension - 2)))
            
        return X
    
    def log_pdf(self, X):
        #assert(shape(X)[1] == 2)
        transformed = X.copy()
        transformed[:, 1] = X[:, 1] - self.bananicity * ((X[:, 0] ** 2) - self.V)
        transformed[:, 0] = X[:, 0] / sqrt(self.V)
        phi = Gaussian(zeros([self.dimension]), eye(self.dimension))
        return phi.log_pdf(transformed)
    
    def emp_quantiles(self, X, quantiles=arange(0.1, 1, 0.1)):
        transformed = X.copy()
        transformed[:, 1] = X[:, 1] - self.bananicity * ((X[:, 0] ** 2) - self.V)
        transformed[:, 0] = X[:, 0] / sqrt(self.V)
        phi = Gaussian(zeros([self.dimension]), eye(self.dimension))
        return phi.emp_quantiles(transformed, quantiles)
    
    def get_plotting_bounds(self):
        if self.bananicity == 0.03 and self.V == 100.0 and self.dimension == 2:
            return [(-20, 20), (-5, 10)]
        else:
            return Distribution.get_plotting_bounds(self)

if __name__ == '__main__':
    banana = Banana(dimension=10)
    X = banana.sample(10000)
    print banana.emp_quantiles(X)
#    Visualise.visualise_distribution(banana)
