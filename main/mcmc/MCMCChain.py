"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from numpy.ma.core import zeros

class MCMCChain(object):
    def __init__(self, mcmc_hammer, mcmc_params):
        self.mcmc_sampler = mcmc_hammer
        self.mcmc_params = mcmc_params
        self.mcmc_outputs = []
        self.is_initialised = False
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "mcmc_hammer="+ str(self.mcmc_sampler)
        s += ", mcmc_params="+ str(self.mcmc_params)
        s += ", mcmc_outputs="+ str(self.mcmc_outputs)
        s += "]"
        return s
    
    def init(self):
        # fields for the chain
        num_iterations = self.mcmc_params.num_iterations
        self.samples = zeros((num_iterations, self.mcmc_sampler.distribution.dimension))
        self.ratios = zeros(num_iterations)
        self.log_liks = zeros(num_iterations)
        self.accepteds = zeros(num_iterations)
        
        # state of the chain
        self.iteration = 0
        
        # init output instances
        for out in self.mcmc_outputs:
            out.prepare()
        
        # init sampler with starting point
        self.mcmc_sampler.init(self.mcmc_params.start.copy())
        self.is_initialised = True
        
    
    def has_finished(self):
        return self.iteration >= self.mcmc_params.num_iterations - 1
    
    def append_mcmc_output(self, output):
        self.mcmc_outputs.append(output)
    
    def run(self):
        if not self.is_initialised:
            self.init()
        
        # run chain
        while self.iteration < self.mcmc_params.num_iterations:
            i = self.iteration
            # store old proposal_object for outputs
            """
            there is a problem here: as Q is first
            initialised in the step() method,
            on the first call Q_old is always None
            -dino
            """
            # mcmc step
            step_output = self.mcmc_sampler.step()
            
            # output updated state
            for out in self.mcmc_outputs:
                out.update(self, step_output)

            # collect results
            self.samples[i, :] = step_output.sample.samples
            self.ratios[i] = step_output.log_ratio
            self.accepteds[i] = step_output.accepted
            self.log_liks[i] = step_output.log_lik
            
            # adapt sampler
            self.mcmc_sampler.adapt(self, step_output)
            
            # adapt chain state
            self.iteration += 1
