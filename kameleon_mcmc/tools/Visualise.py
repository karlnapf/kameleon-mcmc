"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from kameleon_mcmc.distribution.Gaussian import Gaussian
from matplotlib.patches import Ellipse
from matplotlib.pyplot import imshow, ylim, xlim, contour, plot, hold, gca
from numpy import linspace
from numpy.linalg.linalg import eigh
from numpy import zeros, array, exp, arctan2, sqrt
import numpy

class Visualise(object):
    def __init__(self):
        pass
    
    @staticmethod
    def get_plotting_arrays(distribution):
        bounds = distribution.get_plotting_bounds()
        assert(len(bounds) == 2)
        Xs = linspace(bounds[0][0], bounds[0][1])
        Ys = linspace(bounds[1][0], bounds[1][1])
        return Xs, Ys
    
    @staticmethod
    def visualise_distribution(distribution, Z=None, log_density=False, Xs=None, Ys=None):
        """
        Plots the density of a given Distribution instance and plots some
        samples on top.
        """
        if Xs is None or Ys is None:
            Xs, Ys = Visualise.get_plotting_arrays(distribution)
        
        Visualise.plot_density(distribution, Xs, Ys)
        
        if Z is not None:
            hold(True)
            Visualise.plot_data(Z)
            hold(False)
    
    @staticmethod
    def plot_density(distribution, Xs, Ys, log_domain=False):
        """
        Plots a 2D density
        
        density - density - distribution instance to plot
        Xs - x values the density is evaluated at
        Ys - y values the density is evaluated at
        log_domain - if False, density will be put into exponential function
        """
        assert(distribution.dimension == 2)
        
        D = zeros((len(Xs), len(Ys)))
    
        # compute log-density
        for i in range(len(Xs)):
            for j in range(len(Ys)):
                x = array([[Xs[i], Ys[j]]])
                D[j, i] = distribution.log_pdf(x)
        
        if log_domain == False:
            D = exp(D)
        
        im = imshow(D, origin='lower')
        im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
        im.set_interpolation('nearest')
        im.set_cmap('gray')
        ylim([Ys.min(), Ys.max()])
        xlim([Xs.min(), Xs.max()])
      
    @staticmethod  
    def contour_plot_density(distribution, Xs=None, Ys=None, log_domain=False):
        """
        Contour-plots a 2D density. If Gaussian, plots 1.96 interval contour only
        
        density - distribution instance to plot
        Xs - x values the density is evaluated at
        Ys - y values the density is evaluated at
        log_domain - if False, density will be put into exponential function
        """
        if isinstance(distribution, Gaussian) and log_domain == False:
            gca().add_artist(Visualise.get_gaussian_ellipse_artist(distribution))
            gca().plot(distribution.mu[0], distribution.mu[1], 'r*', \
                     markersize=3.0, markeredgewidth=.1)
            return
        
        assert(distribution.dimension == 2)

        if Xs is None:
            (xmin, xmax), _ = distribution.get_plotting_bounds()
            Xs = linspace(xmin, xmax)
           
        if Ys is None:
            _, (ymin, ymax) = distribution.get_plotting_bounds()
            Ys = linspace(ymin, ymax) 
        
        D = zeros((len(Ys), len(Xs)))
        
        # compute log-density
        for i in range(len(Xs)):
            for j in range(len(Ys)):
                x = array([[Xs[i], Ys[j]]])
                D[j, i] = distribution.log_pdf(x)
        
        if log_domain == False:
            D = exp(D)
        
        contour(Xs, Ys, D, origin='lower')
        
    @staticmethod
    def plot_array(Xs, Ys, D):
        """
        Plots a 2D array
        
        Xs - x values the density is evaluated at
        Ys - y values the density is evaluated at
        D - array to plot
        """
        im = imshow(D, origin='lower')
        im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
        im.set_interpolation('nearest')
        im.set_cmap('gray')
        ylim([Ys.min(), Ys.max()])
        xlim([Xs.min(), Xs.max()])

    @staticmethod
    def plot_data(Z, y=None):
        """
        Plots collection of 2D points and optionally adds a marker to one of them
        
        Z - set of row-vectors points to plot
        y - one point that is marked in red, might be None
        """
        plot(Z[:, 0], Z[:, 1], '*', markersize=3.0, markeredgewidth=.1)
        
        if y is not None:
            plot(y[0, 0], y[0, 1], 'r*', markersize=10.0, markeredgewidth=.1)

    @staticmethod
    def get_gaussian_ellipse_artist(gaussian, nstd=1.96, linewidth=1):
        """
        Returns an allipse artist for nstd times the standard deviation of this
        Gaussian
        """
        assert(isinstance(gaussian, Gaussian))
        assert(gaussian.dimension == 2)
        
        # compute eigenvalues (ordered)
        vals, vecs = eigh(gaussian.L.dot(gaussian.L.T))
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        
        theta = numpy.degrees(arctan2(*vecs[:, 0][::-1]))

        # width and height are "full" widths, not radius
        width, height = 2 * nstd * sqrt(vals)
        e = Ellipse(xy=gaussian.mu, width=width, height=height, angle=theta, \
                   edgecolor="red", fill=False, linewidth=linewidth)
        
        return e
