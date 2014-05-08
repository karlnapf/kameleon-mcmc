"""
Copyright (c) 2013-2014 Heiko Strathmann
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 *
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 *
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the author.
"""

from matplotlib.pyplot import subplot, plot,title, show, \
    draw, clf, figure, suptitle, ion, ylim
from numpy import mean
from numpy.ma.core import sqrt

from kameleon_mcmc.mcmc.output.Output import Output


class DiscretePlottingOutput(Output):
    def __init__(self, plot_from=0, lag=1):
        ion()
        self.plot_from = plot_from
        self.lag = lag
    
    def update(self, mcmc_chain, step_output):
        if mcmc_chain.iteration > self.plot_from and mcmc_chain.iteration % self.lag == 0:
            
            # plot "traces"
            num_plots = mcmc_chain.mcmc_sampler.distribution.dimension
            samples = mcmc_chain.samples[0:mcmc_chain.iteration]
            likelihoods = mcmc_chain.log_liks[0:mcmc_chain.iteration]
            num_y = round(sqrt(num_plots))
            num_x = num_plots / num_y + 1
            for i in range(num_plots):
                subplot(num_y, num_x, i + 1)
                plot(samples[:, i], 'b.')
                ylim([-0.2, 1.2])
                title("Trace $x_" + str(i) + "$. Mean: %f" % mean(samples[:, i]))
                
            subplot(num_y, num_x, num_plots + 1)
            plot(likelihoods)
            title("Log-Likelihood")
                
            suptitle(mcmc_chain.mcmc_sampler.__class__.__name__)
            show()
            draw()
            clf()
    
    def prepare(self):
        figure(figsize=(18, 10))
