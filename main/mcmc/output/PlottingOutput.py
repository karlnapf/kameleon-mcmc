"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from main.mcmc.output.Output import Output
from main.tools.Visualise import Visualise
from matplotlib.cm import get_cmap
from matplotlib.pyplot import subplot, plot, xlabel, ylabel, title, hist, show, \
    draw, clf, figure, axis, suptitle
from numpy.core.numeric import zeros
from numpy.ma.core import array, exp, sqrt

class PlottingOutput(Output):
    def __init__(self, distribution=None, plot_from=0, lag=1):
        self.distribution=distribution
        self.plot_from = plot_from
        self.lag=lag
        
        if distribution is not None:
            self.Xs, self.Ys=Visualise.get_plotting_arrays(distribution)
    
    def update(self, mcmc_chain, step_output):
        if mcmc_chain.iteration > self.plot_from and mcmc_chain.iteration%self.lag==0:
            if mcmc_chain.mcmc_sampler.distribution.dimension==2:
                subplot(2, 3, 1)
                if self.distribution is not None:
                    Visualise.plot_array(self.Xs, self.Ys, self.P)
                
                samples=mcmc_chain.samples[0:mcmc_chain.iteration]
                likelihoods=mcmc_chain.log_liks[0:mcmc_chain.iteration]
                proposal_1d=step_output.proposal_object.samples[0,:]
                
                y = samples[len(samples) - 1]
                
                # plot samples, coloured by likelihood
                normalised=likelihoods.copy()
                normalised=normalised-normalised.min()
                normalised=normalised/normalised.max()
                
                cm=get_cmap("jet")
                for i in range(len(samples)):
                    color = cm(normalised[i])
                    plot(samples[i,0], samples[i,1]  ,"o", color=color)       
                
                plot(y[0], y[1], 'r*', markersize=15.0)
                plot(proposal_1d[0], proposal_1d[1], 'y*', markersize=15.0)
                if self.distribution is not None:
                    Visualise.contour_plot_density(mcmc_chain.mcmc_sampler.Q, self.Xs, \
                                                   self.Ys, log_domain=False)
                else:
                    Visualise.contour_plot_density(mcmc_chain.mcmc_sampler.Q)
                    axis('equal')
                
                xlabel("$x_1$")
                ylabel("$x_2$")
                title("Samples")
            
                subplot(2, 3, 2)
                plot(samples[:, 0], 'b')
                title("Trace $x_1$")
                
                subplot(2, 3, 3)
                plot(samples[:, 1], 'b')
                title("Trace $x_2$")
                
                subplot(2, 3, 4)
                plot(mcmc_chain.log_liks[0:mcmc_chain.iteration], 'b')
                title("Log-likelihood")
                
                if len(samples) > 2:
                    subplot(2, 3, 5)
                    hist(samples[:, 0])
                    title("Histogram $x_1$")
            
                    subplot(2, 3, 6)
                    hist(samples[:, 1])
                    title("Histogram $x_2$")
            else:
                # if target dimension is not two, plot traces
                num_plots=mcmc_chain.mcmc_sampler.distribution.dimension
                samples=mcmc_chain.samples[0:mcmc_chain.iteration]
                likelihoods=mcmc_chain.log_liks[0:mcmc_chain.iteration]
                num_y=round(sqrt(num_plots))
                num_x=num_plots/num_y+1
                for i in range(num_plots):
                    subplot(num_y, num_x, i+1)
                    plot(samples[:, i], 'b')
                    title("Trace $x_" +str(i) + "$")
                    
                subplot(num_y, num_x, num_plots+1)
                plot(likelihoods)
                title("Log-Likelihood")
                
            suptitle(mcmc_chain.mcmc_sampler.__class__.__name__)
            show(block=False)
            draw()
            clf()
    
    def prepare(self):
        figure(figsize=(18, 10))
        
        if self.distribution is not None:
            self.P = zeros((len(self.Xs), len(self.Ys)))
            for i in range(len(self.Xs)):
                for j in range(len(self.Ys)):
                    x = array([[self.Xs[i], self.Ys[j]]])
                    self.P[j, i] = self.distribution.log_pdf(x)
            
            self.P = exp(self.P)
