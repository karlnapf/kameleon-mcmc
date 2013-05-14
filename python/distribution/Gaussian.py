from classes.distribution.Distribution import Distribution
from classes.tools.MatrixTools import MatrixTools
from classes.tools.Visualise import Visualise
from numpy.dual import cholesky
from numpy.lib.twodim_base import eye, diag
from numpy.ma.core import array, shape, log, zeros
from numpy.matlib import randn
from scipy.constants.constants import pi
from scipy.linalg.basic import solve_triangular

class Gaussian(Distribution):
    def __init__(self, mu=array([0, 0]), Sigma=eye(2), is_cholesky=False, ell=0):
        Distribution.__init__(self, len(Sigma))
        
        self.mu = mu
        self.ell = ell
        if is_cholesky:
            assert(shape(Sigma)[0] == shape(Sigma)[0])
            self.L = Sigma
        else:
            if ell:
                self.L, _, _ = MatrixTools.low_rank_approx(Sigma, ell)
                self.L = self.L.T
            else:
                self.L = cholesky(Sigma)
    
    def sample(self, n=1):
        V = randn(shape(self.L)[1], n)

        # map to our desired Gaussian and transpose to have row-wise vectors
        return self.L.dot(V).T + self.mu
    
    def log_pdf(self, X):
        log_determinant_part = -sum(log(diag(self.L)))
        
        quadratic_parts = zeros((len(X), 1))
        for i in range(len(X)):
            x = X[i] - self.mu
            
            # solve y=K^(-1)x = L^(-T)L^(-1)x
            y = solve_triangular(self.L, x.T, lower=True)
            y = solve_triangular(self.L.T, y, lower=False)
            quadratic_parts[i] = -0.5 * x.dot(y)
            
        const_part = -0.5 * len(self.L) * log(2 * pi)
        
        return const_part + log_determinant_part + quadratic_parts;
    
if __name__ == '__main__':
    Visualise.visualise_distribution(Gaussian())
