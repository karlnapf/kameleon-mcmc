"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from matplotlib.pyplot import savefig, legend, show, figure, \
    axis
from numpy import array
from os import makedirs

from kameleon_mcmc.distribution.Banana import Banana
from kameleon_mcmc.distribution.Flower import Flower
from kameleon_mcmc.distribution.Ring import Ring
from kameleon_mcmc.kernel.GaussianKernel import GaussianKernel
from kameleon_mcmc.kernel.MaternKernel import MaternKernel
from kameleon_mcmc.paper_figures.GFunction import GFunction
import latex_plot_init


# global variables that are used by both functions
print "Generating g-functions"
distributions = [Ring(), Banana(), Flower(), Ring(), Banana(), Flower()]
kernels = [GaussianKernel(.5), GaussianKernel(3), GaussianKernel(2), \
           MaternKernel(1), MaternKernel(2), MaternKernel(1)]
num_samples = [200, 200, 1000, 200, 200, 1000]
nu2s = [0.01, 0.5, 0.2, 0.01, 0.5, 0.2]
g_functions = [GFunction(distributions[i], n=num_samples[i], kernel=kernels[i], \
                         nu2=nu2s[i], gamma=0.1, ell=15, nXs=200, nYs=200) \
                         for i in range(len(distributions))]
ys = [array([2, -3]) , array([2, -3]), array([-2, -8.5]), array([2, -3]) , array([2, -3]), array([-2, -8.5])]
num_proposals = [10, 10, 40, 10, 10, 40]
gradient_scales = [35, 30, 20, 35, 30, 20]
num_betas = [0, 5, 0, 0, 5, 0]

def plot_g_functions():
    for i in range(len(distributions)):
        for j in range(num_betas[i]):
            print distributions[i].__class__.__name__, g_functions[i].kernel.__class__.__name__, j
            figure(figsize=(3, 2))
            g_functions[i].plot(ys[i], gradient_scale=gradient_scales[i], plot_data=True)
            axis('off')
            g_functions[i].resample_beta()
            
            if j is 0:
                legend(["Samples $\{z_i\}_{i=1}^{" + str(num_samples[i]) + "}$", "Current position $y$"], \
                       numpoints=1, loc="upper center")
            
            filename_base = "plots/g_gunction_" + distributions[i].__class__.__name__ + str(j) +\
                            + g_functions[i].kernel.__class__.__name__
            savefig(filename_base + ".eps", bbox_inches='tight')
            savefig(filename_base + ".png", bbox_inches='tight')
        
def plot_proposals():
    for i in range(len(distributions)):
        print distributions[i].__class__.__name__, g_functions[i].kernel.__class__.__name__
        figure(figsize=(3, 3))
        ys = distributions[i].get_proposal_points(num_proposals[i])
        g_functions[i].plot_proposal(ys)
        axis('off')
        
        filename_base="plots/proposals_" + distributions[i].__class__.__name__ + \
                        "_" + g_functions[i].kernel.__class__.__name__
        
        savefig(filename_base + ".eps", bbox_inches='tight')
        savefig(filename_base + ".png", bbox_inches='tight')

if __name__ == '__main__':
    try:
        makedirs("plots/")
    except OSError:
        pass
    
    print "generating g-function plots"
#     plot_g_functions()
    
    print "generating proposal plots"
    plot_proposals()
    show()
    
    
