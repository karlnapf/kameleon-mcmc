"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from kameleon_mcmc.distribution.Ring import Ring
from kameleon_mcmc.gp.GaussianProcess import GaussianProcess
from kameleon_mcmc.gp.LaplaceApproximation import LaplaceApproximation
from kameleon_mcmc.gp.LogitLikelihood import LogitLikelihood
from kameleon_mcmc.gp.SquaredExponentialCovariance import SquaredExponentialCovariance
from matplotlib.pyplot import pcolor, hold, plot, colorbar, clf, subplot, show, \
    figure
from numpy import unravel_index
from numpy.core.function_base import linspace
from numpy.core.shape_base import vstack
from numpy.linalg.linalg import norm
from numpy.ma.core import asarray, reshape, zeros, array, ones, exp
import itertools
import unittest


class LaplaceApproxmationTests(unittest.TestCase):

    def test_mode_newton_2d(self):
        X = asarray([-1, 1])
        X = reshape(X, (len(X), 1))
        y = asarray([+1 if x >= 0 else -1 for x in X])
        covariance = SquaredExponentialCovariance(sigma=1, scale=1)
        likelihood = LogitLikelihood()
        gp = GaussianProcess(y, X, covariance, likelihood)
        laplace = LaplaceApproximation(gp, newton_start=asarray([3, 3]))
        
        f_mode, _, steps = laplace.find_mode_newton(return_full=True)
        F = linspace(-10, 10, 20)
        D = zeros((len(F), len(F)))
        for i in range(len(F)):
            for j in range(len(F)):
                f = asarray([F[i], F[j]])
                D[i, j] = gp.log_posterior_unnormalised(f)
           
        idx = unravel_index(D.argmax(), D.shape)
        empirical_max = asarray([F[idx[0]], F[idx[1]]])
        
        pcolor(F, F, D)
        hold(True)
        plot(steps[:, 0], steps[:, 1])
        plot(f_mode[1], f_mode[0], 'mo', markersize=10)
        hold(False)
        colorbar()
        clf()
#        show()
           
        self.assertLessEqual(norm(empirical_max - f_mode), 1)
        

    def test_get_gaussian_2d(self):
        X = asarray([-1, 1])
        X = reshape(X, (len(X), 1))
        y = asarray([+1 if x >= 0 else -1 for x in X])
        covariance = SquaredExponentialCovariance(sigma=1, scale=1)
        likelihood = LogitLikelihood()
        gp = GaussianProcess(y, X, covariance, likelihood)
        laplace = LaplaceApproximation(gp, newton_start=asarray([3, 3]))
        
        f_mode, L, steps = laplace.find_mode_newton(return_full=True)
        gaussian = laplace.get_gaussian(f_mode, L)
        F = linspace(-10, 10, 20)
        D = zeros((len(F), len(F)))
        Q = array(D, copy=True)
        for i in range(len(F)):
            for j in range(len(F)):
                f = asarray([F[i], F[j]])
                D[i, j] = gp.log_posterior_unnormalised(f)
                Q[i, j] = gaussian.log_pdf(f.reshape(1, len(f)))
        
        subplot(1, 2, 1)
        pcolor(F, F, D)
        hold(True)
        plot(steps[:, 0], steps[:, 1])
        plot(f_mode[1], f_mode[0], 'mo', markersize=10)
        hold(False)
        colorbar()
        subplot(1, 2, 2)
        pcolor(F, F, Q)
        hold(True)
        plot(f_mode[1], f_mode[0], 'mo', markersize=10)
        hold(False)
        colorbar()
#        show()
        clf()
        
    def test_predict(self):
        # define some easy training data and predict predictive distribution
        circle1 = Ring(variance=1, radius=3)
        circle2 = Ring(variance=1, radius=10)
        
        n = 100
        X = circle1.sample(n / 2).samples
        X = vstack((X, circle2.sample(n / 2).samples))
        y = ones(n)
        y[:n / 2] = -1.0
        
#        plot(X[:n/2,0], X[:n/2,1], 'ro')
#        hold(True)
#        plot(X[n/2:,0], X[n/2:,1], 'bo')
#        hold(False)
#        show()

        covariance = SquaredExponentialCovariance(1, 1)
        likelihood = LogitLikelihood()
        gp = GaussianProcess(y, X, covariance, likelihood)

        # predict on mesh
        n_test = 20
        P = linspace(X[:, 0].min() - 1, X[:, 1].max() + 1, n_test)
        Q = linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, n_test)
        X_test = asarray(list(itertools.product(P, Q)))
#        Y_test = exp(LaplaceApproximation(gp).predict(X_test).reshape(n_test, n_test))
        Y_train = exp(LaplaceApproximation(gp).predict(X))
        print Y_train
        
        print Y_train>0.5
        print y
        
#        pcolor(P, Q, Y_test)
#        colorbar()
#        show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
