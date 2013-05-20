from main.distribution.Distribution import Distribution
from main.tools.MatrixTools import MatrixTools
from main.tools.Visualise import Visualise
from numpy.dual import cholesky, norm, eig
from numpy.lib.twodim_base import eye, diag
from numpy.ma.core import array, shape, log, zeros, arange, mean, ones
from numpy.random import randn
from scipy.constants.constants import pi
from scipy.linalg.basic import solve_triangular
from scipy.stats.distributions import chi2

class Gaussian(Distribution):
    def __init__(self, mu=array([0, 0]), Sigma=eye(2), is_cholesky=False, ell=0):
        Distribution.__init__(self, len(Sigma))
        
        self.mu = mu
        
        assert(ell>=0)
        if ell:
            self.ell = ell
            
        if is_cholesky:
            assert(shape(Sigma)[0] == shape(Sigma)[0])
            self.L = Sigma
            self.Sigma=None
        else:
            self.Sigma=Sigma
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
    
    def emp_quantiles(self,X,quantiles=arange(0.1,1,0.1)):
        #need inverse chi2 cdf with self.dimension degrees of freedom
        chi2_instance=chi2(self.dimension)
        cutoffs=chi2_instance.isf(1-quantiles)
        #whitening
        D,U = eig(self.Sigma)
        D = D ** (-0.5)
        W=(diag(D).dot(U.T).dot((X-self.mu).T)).T
        norms_squared = array([norm(w)**2 for w in W])
        results = zeros([len(quantiles)])
        for jj in range(0,len(quantiles)):
            results[jj] = mean(norms_squared<cutoffs[jj])
        return results
    
    
if __name__ == '__main__':
    mu=array([5,2])
    Sigma=eye(2)
    Sigma[0,0]=20
    R=MatrixTools.rotation_matrix(pi/4)
    Sigma=R.dot(Sigma).dot(R.T)
    gaussian_instance=Gaussian(mu, Sigma)
    X=gaussian_instance.sample(1000)
    print gaussian_instance.emp_quantiles(X)
    Visualise.visualise_distribution(gaussian_instance)
