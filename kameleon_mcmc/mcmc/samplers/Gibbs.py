"""
Copyright (c) 2014 Heiko Strathmann, Dino Sejdinovic
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
from kameleon_mcmc.distribution.full_conditionals.FullConditionals import FullConditionals
from kameleon_mcmc.mcmc.samplers.MCMCSampler import MCMCSampler


class Gibbs(MCMCSampler):
    """
    Gibbs sampler for arbitrary sets of full conditional distributions
    """
    def __init__(self, full_conditionals):
        if not isinstance(full_conditionals, FullConditionals):
            raise TypeError("Gibbs require full conditional distribution instance")
        
        MCMCSampler.__init__(self, full_conditionals)
        
        # pdf is constant, and therefore symmetric
        self.is_symmetric = True
        
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += MCMCSampler.__str__(self)
        s += "]"
        return s
    
    def construct_proposal(self, y):
        """
        Returns a distribution object that represents the current full
        conditional of one random variable
        """
        
        return self.distribution
        
    
    def adapt(self, mcmc_chain, step_output):
        """
        Nothing for this one since conditionals are fixed
        """
        pass
