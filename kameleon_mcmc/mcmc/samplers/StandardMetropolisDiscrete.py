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

from kameleon_mcmc.distribution.Distribution import Distribution
from kameleon_mcmc.distribution.proposals.DiscreteRandomWalkProposal import DiscreteRandomWalkProposal
from kameleon_mcmc.mcmc.samplers.MCMCSampler import MCMCSampler


class StandardMetropolisDiscrete(MCMCSampler):
    """
    A random walk on the hypercube. Flips a random number of components.
    """
    def __init__(self, distribution, spread, flip_at_least_one=True):
        if not isinstance(distribution, Distribution):
            raise TypeError("Target must be a Distribution object")
        
        if not type(spread) is float:
            raise TypeError("Spread must be a float")
        
        if not (spread > 0. and spread < 1.):
            raise ValueError("Spread must be a probability")
        
        MCMCSampler.__init__(self, distribution)
        
        self.spread = spread
        self.flip_at_least_one = flip_at_least_one
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "spread=" + str(self.spread)
        s += ", " + MCMCSampler.__str__(self)
        s += "]"
        return s
    
    def construct_proposal(self, y):
        return DiscreteRandomWalkProposal(y, self.spread, self.flip_at_least_one)
    
    def adapt(self, mcmc_chain, step_output):
        """
        Nothing for this one since it is not adaptive
        """
        pass
    
