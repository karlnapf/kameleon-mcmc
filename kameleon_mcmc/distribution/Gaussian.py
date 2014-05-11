"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from kameleon_mcmc.distribution.Distribution import Distribution, Sample
from kameleon_mcmc.tools.MatrixTools import MatrixTools
from numpy.lib.twodim_base import eye, diag
from numpy.linalg import cholesky, norm, eig
from numpy import array, shape, log, zeros, arange, mean
from numpy.random import randn
from scipy.constants.constants import pi
from scipy.linalg.basic import solve_triangular
from scipy.stats.distributions import chi2
from numpy.linalg.linalg import LinAlgError

class Gaussian(Distribution):
    def __init__(self, mu=array([0, 0]), Sigma=eye(2), is_cholesky=False, ell=None):
        Distribution.__init__(self, len(Sigma))
        
        assert(len(shape(mu)) == 1)
        assert(max(shape(Sigma)) == len(mu))
        self.mu = mu
        self.ell = ell
        if is_cholesky: 
            self.L = Sigma
            if ell == None:
                assert(shape(Sigma)[0] == shape(Sigma)[1])
            else:
                assert(shape(Sigma)[1] == ell)
        else: 
            assert(shape(Sigma)[0] == shape(Sigma)[1])
            if ell is not None:
                self.L, _, _ = MatrixTools.low_rank_approx(Sigma, ell)
                self.L = self.L.T
                assert(shape(self.L)[1] == ell)
            else:
                try:
                    self.L = cholesky(Sigma)
                except LinAlgError:
                    # some really crude check for PSD (which only corrects for orunding errors
                    self.L = cholesky(Sigma+eye(len(Sigma))*1e-5)
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "mu=" + str(self.mu)
        s += ", L=" + str(self.L)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s
    
    def sample(self, n=1):
        if self.ell is None:
            V = randn(self.dimension, n)
        else:
            V = randn(self.ell, n)

        # map to our desired Gaussian and transpose to have row-wise vectors
        return Sample(self.L.dot(V).T + self.mu)
        
    def log_pdf(self, X):
        assert(len(shape(X)) == 2)
        assert(shape(X)[1] == self.dimension)
        
        log_determinant_part = -sum(log(diag(self.L)))
        
        quadratic_parts = zeros(len(X))
        for i in range(len(X)):
            x = X[i] - self.mu
            
            # solve y=K^(-1)x = L^(-T)L^(-1)x
            y = solve_triangular(self.L, x.T, lower=True)
            y = solve_triangular(self.L.T, y, lower=False)
            quadratic_parts[i] = -0.5 * x.dot(y)
            
        const_part = -0.5 * len(self.L) * log(2 * pi)
        
        return const_part + log_determinant_part + quadratic_parts
    
    def log_pdf_at_quantile(self, alphas):
        """
        Computes the log-pdf at a given 1d-vector of quantiles
        """
        chi2_instance = chi2(self.dimension)
        cuttoffs = chi2_instance.isf(1 - alphas)
        
        log_determinant_part = -sum(log(diag(self.L)))
        quadratic_part = -0.5 * cuttoffs
        const_part = -0.5 * len(self.L) * log(2 * pi)
        
        return const_part + log_determinant_part + quadratic_part
    
    def emp_quantiles(self, X, quantiles=arange(0.1, 1, 0.1)):
        # need inverse chi2 cdf with self.dimension degrees of freedom
        chi2_instance = chi2(self.dimension)
        cutoffs = chi2_instance.isf(1 - quantiles)
        # whitening
        D, U = eig(self.L.dot(self.L.T))
        D = D ** (-0.5)
        W = (diag(D).dot(U.T).dot((X - self.mu).T)).T
        norms_squared = array([norm(w) ** 2 for w in W])
        results = zeros([len(quantiles)])
        for jj in range(0, len(quantiles)):
            results[jj] = mean(norms_squared < cutoffs[jj])
        return results
