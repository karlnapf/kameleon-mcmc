"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from main.distribution.Banana import Banana
from main.distribution.Constant import Constant
from main.distribution.Discrete import Discrete
from main.distribution.Gaussian import Gaussian
from main.gp.GaussianProcess import GaussianProcess
from main.gp.LaplaceApproximation import LaplaceApproximation
from main.gp.LogitLikelihood import LogitLikelihood
from main.gp.SquaredExponentialCovariance import SquaredExponentialCovariance
from matplotlib import cm
from matplotlib.pyplot import imshow, show, figure, plot, hold, scatter
from numpy.core.function_base import linspace
from numpy.core.numeric import inf
from numpy.core.shape_base import hstack, vstack
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import cholesky
from numpy.ma.core import zeros, exp, ones, log, asarray, prod, shape, array
from numpy.random import randn, randint
from scipy import integrate
from scipy.linalg.basic import solve_triangular

class GPToyData(object):
    """
    Samples from a 2D classification GP with a distribution over hyper-parameters
    """

    def __init__(self, theta_distribution):
        self.theta_distribution = theta_distribution
        
        
    def sample_two_gaussians(self, n=1):
        
        # sample hyper-parameter distribution
        thetas = self.theta_distribution.sample(n).samples
        print thetas
        
        data = zeros((n, 2))
        labels = zeros(n)
        
        # fix covariate distributions and labels, XOR like Gaussian mixture
        scale=3
        gaussians=[]
        gaussians.append(Gaussian(asarray([1,1])*scale, eye(2)))
        gaussians.append(Gaussian(asarray([-1,1])*scale, eye(2)))
        gaussians.append(Gaussian(asarray([-1,-1])*scale, eye(2)))
        gaussians.append(Gaussian(asarray([1,-1])*scale, eye(2)))
        
        # sample covariates
        self.covariates=zeros((0,2))
        self.labels=y=zeros(0)
        for i in range(4):
            X=gaussians[i].sample(n/4).samples
            y=ones(n/4)
            
            label=prod(gaussians[i].mu)<0
            if label:
                y=-y
                
            self.covariates=vstack((self.covariates, X))
            self.labels=hstack((self.labels, y))
        
        # sample a point for each theta, apply GP to generate label
        for i in range(n):
            theta = thetas[i]
            
            # GP for current hyper-parameters
            covariance = SquaredExponentialCovariance(exp(theta[0]), exp(theta[1]))
            likelihood=LogitLikelihood()
            gp=GaussianProcess(self.labels, self.covariates, covariance, likelihood)

            # pick a random point
            x=randn(1,2)
            labels[i]=LaplaceApproximation(gp).predict(x)
            data[i,:]=x
            
        return data, labels
        
if __name__ == '__main__':
    theta_distribution = Banana()
    theta_distribution=Constant()
    gp_data = GPToyData(theta_distribution)
    x, y = gp_data.sample(15)
    print x, y
    idx_a = y == 1.0
    idx_b = y == -1.0
    plot(x[idx_a, 0], x[idx_a, 1], 'ro')
    plot(x[idx_b, 0], x[idx_b, 1], 'bo')
    show()
