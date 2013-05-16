from main.mcmc.output.Output import Output
from main.tools.Visualise import Visualise
from matplotlib.pyplot import subplot, plot, xlabel, ylabel, title, hist, show, \
    draw, clf, figure
from numpy.core.numeric import zeros
from numpy.ma.core import array, exp

class PlottingOutput(Output):
    def __init__(self, distribution, plot_from=0):
        self.distribution=distribution
        self.plot_from = plot_from
        self.Xs, self.Ys=Visualise.get_plotting_arrays(distribution)
    
    def update(self, mcmc_params, proposal, samples, log_liks, Q):
        if len(samples) > self.plot_from:
            subplot(2, 3, 1)
            Visualise.plot_array(self.Xs, self.Ys, self.P)
            
            y = samples[len(samples) - 1]
            plot(samples[:, 0], samples[:, 1], 'm')
            plot(y[0], y[1], 'r*', markersize=15.0)
            plot(proposal[0, 0], proposal[0, 1], 'y*', markersize=15.0)
            Visualise.contour_plot_density(Q, self.Xs, self.Ys, log_domain=False)
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
            plot(log_liks, 'b')
            title("Log-likelihood")
            
            if len(samples) > 2:
                subplot(2, 3, 5)
                hist(samples[:, 0])
                title("Histogram $x_1$")
        
                subplot(2, 3, 6)
                hist(samples[:, 1])
                title("Histogram $x_2$")
                
            show(block=False)
            draw()
            clf()
    
    def prepare(self):
        figure(figsize=(20, 13))
        self.P = zeros((len(self.Xs), len(self.Ys)))
        for i in range(len(self.Xs)):
            for j in range(len(self.Ys)):
                x = array([[self.Xs[i], self.Ys[j]]])
                self.P[j, i] = self.distribution.log_pdf(x)
        
        self.P = exp(self.P)
