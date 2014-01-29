"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from matplotlib import use
from matplotlib.pyplot import plot, fill_between, savefig, ylim, clf, title, \
    ioff, close, figure
from numpy import vstack
from numpy.lib.npyio import savetxt
from numpy.linalg.linalg import norm
from numpy.ma.core import arange, zeros, mean, std, allclose, sqrt, asarray, \
    array
from numpy.ma.extras import median

from main.experiments.ExperimentAggregator import ExperimentAggregator
from main.kernel.GaussianKernel import GaussianKernel
from main.tools.RCodaTools import RCodaTools


use('Agg')


class GroundTruthSingleChainExperimentAggregator(ExperimentAggregator):
    def __init__(self, folders, thinning_factor=1):
        ExperimentAggregator.__init__(self, folders)
        self.thinning_factor=thinning_factor
    
    def __process_results__(self):
        lines = []
        if len(self.experiments) == 0:
            lines.append("no experiments to process")
            return
        
        # burnin and dimension are the same for all chains
        burnin = self.experiments[0].mcmc_chain.mcmc_params.burnin
        dim = self.experiments[0].mcmc_chain.mcmc_sampler.distribution.dimension
        
        # collect all thinned samples of all chains in here
        merged_samples=zeros((0, dim))
        
        for i in range(len(self.experiments)):
            lines.append("Processing chain %d" % i)
            
            # discard samples before burn in
            lines.append("Discarding burnin of %d" % burnin)
            burned_in = self.experiments[i].mcmc_chain.samples[burnin:, :]
            
            # thin out by factor and store thinned samples
            indices=arange(0, len(burned_in), self.thinning_factor)
            lines.append("Thinning by factor of %d, giving %d samples" \
                         % (self.thinning_factor, len(indices)))
            thinned=burned_in[indices, :]
            merged_samples=vstack((merged_samples, thinned))

        # dump merged samples to disc
        fname=self.experiments[0].name + "_merged_samples.txt"
        lines.append("Storing %d samples in file %s" % (len(merged_samples), fname))
        savetxt(fname, merged_samples)

        return lines

    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "ref_quantiles=" + str(self.ref_quantiles)
        s += ", " + ExperimentAggregator.__str__(self)
        s += "]"
        return s
