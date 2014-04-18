"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from kameleon_mcmc.gp.GaussianProcess import GaussianProcess
from kameleon_mcmc.gp.covariance.SquaredExponentialCovariance import \
    SquaredExponentialCovariance
from kameleon_mcmc.gp.likelihood.LogitLikelihood import LogitLikelihood
from numpy.linalg.linalg import norm
from numpy.ma.core import asarray
from numpy.random import randn, randint
import unittest

class GaussianProcessTests(unittest.TestCase):

    def test_log_lik_multiple1(self):
        n=3
        y=randint(0,2,n)*2-1
        f=randn(n)
        
        X=randn(n,2)
        covariance=SquaredExponentialCovariance(sigma=1, scale=1)
        likelihood=LogitLikelihood()
        gp=GaussianProcess(y, X, covariance, likelihood)
        
        single = gp.log_likelihood(f)
        multiple=gp.log_likelihood_multiple(f.reshape(1,n))
        
        self.assertLessEqual(norm(single-multiple), 1e-10)
    
    def test_log_lik_multiple2(self):
        n=3
        y=randint(0,2,n)*2-1
        F=randn(10,n)
        
        X=randn(n,2)
        covariance=SquaredExponentialCovariance(sigma=1, scale=1)
        likelihood=LogitLikelihood()
        gp=GaussianProcess(y, X, covariance, likelihood)
        
        singles = asarray([gp.log_likelihood(f) for f in F])
        multiples=gp.log_likelihood_multiple(F)
        
        self.assertLessEqual(norm(singles-multiples), 1e-10)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()