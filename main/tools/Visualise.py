from matplotlib.pyplot import imshow, ylim, xlim, contour, plot, hold
from numpy.core.function_base import linspace
from numpy.ma.core import zeros, array, exp

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
    def visualise_distribution(distribution, Z=None, log_density=False):
        """
        Plots the density of a given Distribution instance and plots some
        samples on top.
        """
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
    def contour_plot_density(distribution, Xs, Ys, levels=None, log_domain=False):
        """
        Contour-plots a 2D density
        
        density - distribution instance to plot
        Xs - x values the density is evaluated at
        Ys - y values the density is evaluated at
        log_domain - if False, density will be put into exponential function
        """
        assert(distribution.dimension == 2)
        
        D = zeros((len(Ys), len(Xs)))
        
        # compute log-density
        for i in range(len(Xs)):
            for j in range(len(Ys)):
                x = array([[Xs[i], Ys[j]]])
                D[j, i] = distribution.log_pdf(x)
        
        if log_domain == False:
            D = exp(D)
        
        if levels is None:
            contour(Xs, Ys, D, origin='lower')
        else:
            contour(Xs, Ys, D, levels=levels, origin='lower', linewidth=3)
        
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
        plot(Z[:, 0], Z[:, 1], '*', markersize=5.0)
        
        if y is not None:
            plot(y[0, 0], y[0, 1], 'r*', markersize=15.0)
