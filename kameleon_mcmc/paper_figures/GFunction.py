"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from matplotlib.pyplot import hold, quiver, draw
from numpy import reshape, array, shape, meshgrid, zeros, linspace

from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
from kameleon_mcmc.kernel.Kernel import Kernel
from kameleon_mcmc.mcmc.samplers.Kameleon import Kameleon
from kameleon_mcmc.tools.Visualise import Visualise


class GFunction(object):
    def __init__(self, distribution, n=200, kernel=GaussianKernel(3), nu2=0.1, \
                 gamma=0.1, ell=15, nXs=100, nYs=100):
        self.kernel = kernel
        self.distribution = distribution
        self.nu2 = nu2
        self.gamma = gamma
        self.ell = ell
        
        # fix some samples
        self.Z = self.distribution.sample(n).samples
        
        # evaluate and center kernel and scale
        self.K = self.kernel.kernel(self.Z, None)
        self.K = Kernel.center_kernel_matrix(self.K)
        
        # sample beta
        self.rkhs_gaussian = Gaussian(mu=zeros(len(self.Z)), Sigma=self.K, is_cholesky=False, \
                            ell=self.ell)
        self.beta = self.rkhs_gaussian.sample().samples
        
        # plotting resolution
        [(xmin, xmax), (ymin, ymax)] = self.distribution.get_plotting_bounds()
        self.Xs = linspace(xmin, xmax, nXs)
        self.Ys = linspace(ymin, ymax, nYs)
        
    def resample_beta(self):
        self.beta = self.rkhs_gaussian.sample().samples

    def compute(self, x, y, Z, beta):
        """
        Given two points x and y, a set of samples Z, and a vector beta,
        and a kernel function, this computes the g function
        g(x,beta,Z)=||k(x,.)-f|| for f=k(.,y)+sum_i beta_i*k(.,z_i)
                   =k(x,x) -2k(x,y) -2sum_i beta_i*k(x,z_i) +C
        Constant C is not computed
        """
        first = self.kernel.kernel(x, x)
        second = -2 * self.kernel.kernel(x, y)
        third = -2 * self.kernel.kernel(x, Z).dot(beta.T)
        return first + second + third

    def compute_gradient(self, x, y, Z, beta):
        """
        Given two points x and y, a set of samples Z, and a vector beta,
        and a kernel gradient, this computes the g function's gradient
        \nabla_x g(x,beta,Z)=\nabla_x k(x,x) -2k(x,y) -2sum_i beta_i*k(x,z_i)
        """
        x_2d = reshape(x, (1, len(x)))
        first = self.kernel.gradient(x, x_2d)
        second = -2 * self.kernel.gradient(x, y)
        
        # compute sum_i beta_i \nabla_x k(x,z_i) and beta is a row vector
        gradients = self.kernel.gradient(x, Z)
        third = -2 * beta.dot(gradients)
        
        return first + second + third


    def plot(self, y=array([[-2, -2]]), gradient_scale=None, plot_data=False):
        
        # where to evaluate G?
        G = zeros((len(self.Ys), len(self.Xs)))
    
        # for plotting the gradient field, each U and V are one dimension of gradient
        if gradient_scale is not None:
            GXs2 = linspace(self.Xs.min(), self.Xs.max(), 30)
            GYs2 = linspace(self.Ys.min(), self.Ys.max(), 20)
            X, Y = meshgrid(GXs2, GYs2)
            U = zeros(shape(X))
            V = zeros(shape(Y))
    
        # evaluate g at a set of points in Xs and Ys
        for i in range(len(self.Xs)):
#            print i, "/", len(self.Xs)
            for j in range(len(self.Ys)):
                x_2d = array([[self.Xs[i], self.Ys[j]]])
                y_2d = reshape(y, (1, len(y)))
                G[j, i] = self.compute(x_2d, y_2d, self.Z, self.beta)
    
        # gradient at lower resolution
        if gradient_scale is not None:
            for i in range(len(GXs2)):
#                print i, "/", len(GXs2)
                for j in range(len(GYs2)):
                    x_1d = array([GXs2[i], GYs2[j]])
                    y_2d = reshape(y, (1, len(y)))
                    G_grad = self.compute_gradient(x_1d, y_2d, self.Z, self.beta)
                    U[j, i] = -G_grad[0, 0]
                    V[j, i] = -G_grad[0, 1]
    
        # plot g and Z points and y
        y_2d = reshape(y, (1, len(y)))
        Visualise.plot_array(self.Xs, self.Ys, G)
        
        if gradient_scale is not None:
            hold(True)
            quiver(X, Y, U, V, color='y', scale=gradient_scale)
            hold(False)

        if plot_data:
            hold(True)
            Visualise.plot_data(self.Z, y_2d)
            hold(False)

    def plot_proposal(self, ys):
        # evaluate density itself
        Visualise.visualise_distribution(self.distribution, Z=self.Z, Xs=self.Xs, Ys=self.Ys)
        
        # precompute constants of proposal
        mcmc_hammer = Kameleon(self.distribution, self.kernel, self.Z, \
                                 self.nu2, self.gamma)
        
        # plot proposal around each y
        for y in ys:
            mu, L_R = mcmc_hammer.compute_constants(y)
            gaussian = Gaussian(mu, L_R, is_cholesky=True)
            
            hold(True)
            Visualise.contour_plot_density(gaussian)
            hold(False)
            draw()
