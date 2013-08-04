"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from main.gp.GaussianProcess import GaussianProcess
from main.gp.covariance.SquaredExponentialCovariance import \
    SquaredExponentialCovariance
from main.gp.inference.LaplaceApproximation import LaplaceApproximation
from main.gp.likelihood.LogitLikelihood import LogitLikelihood
from matplotlib.pyplot import plot, show, subplot, hist
from numpy.linalg.linalg import cholesky
from numpy.ma.core import asarray, reshape, exp, cumsum, arange

# create example data and a GP with Laplace approximation
X = asarray([1.0,-1.0])
X = reshape(X, (len(X), 1))
y = asarray([+1. if x >= 0 else -1. for x in X])
covariance = SquaredExponentialCovariance(sigma=2, scale=1)
print covariance.compute(X)
covariance = SquaredExponentialCovariance(sigma=2, scale=2)
print covariance.compute(X)
likelihood = LogitLikelihood()
gp = GaussianProcess(y, X, covariance, likelihood)
laplace = LaplaceApproximation(gp)
proposal=laplace.get_gaussian()

proposal.mu = asarray([0.356148648185266203,-0.356148648185266425])
proposal.L = cholesky(asarray([[0.80268328893646268,    0.087761629500267016],
[    0.087761629500267016,    0.80268328893646268]]))



print gp.log_ml_estimate(proposal=proposal, n=100000)
#
#n=500
#values=[exp(gp.log_ml_estimate(proposal=proposal, n=200)) for _ in range(n)]
#subplot(121)
#plot(cumsum(values)/(arange(len(values))+1))
#subplot(122)
#hist(values)
#show()
