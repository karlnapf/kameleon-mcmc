"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from kameleon_mcmc.gp.GPTools import GPTools
from kameleon_mcmc.gp.GaussianProcess import GaussianProcess
from kameleon_mcmc.gp.covariance.SquaredExponentialCovariance import \
    SquaredExponentialCovariance
from kameleon_mcmc.gp.inference.LaplaceApproximation import LaplaceApproximation
from kameleon_mcmc.gp.likelihood.LogitLikelihood import LogitLikelihood
from numpy.linalg.linalg import norm
from numpy.ma.core import log, exp, asarray, reshape, mean
import unittest


class Test(unittest.TestCase):
    def test_log_sum_exp(self):
        X=asarray([0.1,0.2,0.3,0.4])
        direct=log(sum(exp(X)))
        indirect=GPTools.log_sum_exp(X)
        self.assertLessEqual(norm(direct-indirect), 1e-10)
        
    def test_log_mean_exp(self):
        X = asarray([-1, 1])
        X = reshape(X, (len(X), 1))
        y = asarray([+1. if x >= 0 else -1. for x in X])
        covariance = SquaredExponentialCovariance(sigma=1, scale=1)
        likelihood = LogitLikelihood()
        gp = GaussianProcess(y, X, covariance, likelihood)
        laplace = LaplaceApproximation(gp, newton_start=asarray([3, 3]))
        proposal=laplace.get_gaussian()
        
        n=200
        prior = gp.get_gp_prior()
        samples = proposal.sample(n).samples
        
        log_likelihood=asarray([gp.log_likelihood(f) for f in samples])
        log_prior = prior.log_pdf(samples)
        log_proposal = proposal.log_pdf(samples)
        
        X=log_likelihood+log_prior-log_proposal
        
        a=log(mean(exp(X)))
        b=GPTools.log_mean_exp(X)
        
        self.assertLessEqual(a-b, 1e-5)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()