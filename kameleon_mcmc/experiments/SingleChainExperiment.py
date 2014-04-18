"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from kameleon_mcmc.experiments.Experiment import Experiment
from numpy.ma.core import arange

class SingleChainExperiment(Experiment):
    def __init__(self, mcmc_chain, experiment_dir="", name=None, \
                 ref_quantiles=arange(0.1, 1, 0.1)):

        if name is None:
            name = mcmc_chain.mcmc_sampler.__class__.__name__ + "_" + \
                 mcmc_chain.mcmc_sampler.distribution.__class__.__name__
                 
        self.mcmc_chain = mcmc_chain
        self.ref_quantiles = ref_quantiles
        Experiment.__init__(self, experiment_dir, name)
        
    
    def __run_experiment__(self):
        print "starting mcmc chain"
        self.mcmc_chain.run()
        
        # compute quantiles after burn_in if possible
            
        try:
            print "trying to precompute std quantiles"
            burned_in = self.mcmc_chain.samples[self.mcmc_chain.mcmc_params.burnin:, :]
            self.quantiles = self.mcmc_chain.mcmc_sampler.distribution.emp_quantiles(\
                                  burned_in, self.ref_quantiles)
        except NotImplementedError:
            print "computing quantiles is not possible, skipping"

    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "mcmc_chain=" + str(self.mcmc_chain)
        s += ", " + Experiment.__str__(self)
        s += "]"
        return s
