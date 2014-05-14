"""
Copyright (c) 2013-2014 Heiko Strathmann, Dino Sejdinovic
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

import numpy
from numpy.random import rand

from kameleon_mcmc.distribution.Bernoulli import Bernoulli
from kameleon_mcmc.distribution.full_conditionals.FullConditionals import FullConditionals


class BernoulliFullConditionals(FullConditionals):
    """
    Implements the full conditional distributions for a multivariate Bernoulli.
    Takes an instance of a Bernoulli and computes the full conditionals.
    """
    def __init__(self, full_target, current_state, schedule="in_turns", index_block=None):
        if not isinstance(full_target, Bernoulli):
            raise TypeError("Given full Bernoulli is not a Bernoulli")
        
        FullConditionals.__init__(self, full_target, current_state, schedule, index_block)
        
        
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += FullConditionals.__str__(self)
        s += "]"
        return s
    
    def sample_conditional(self, index):
        if index < 0 or index >= self.dimension:
            raise ValueError("Conditional index out of bounds")
        
        # since all components are independent, just sample from Bernoulli with given index
        return rand(1,) < self.full_target.ps[index]

    def get_current_state_array(self):
        return FullConditionals.get_current_state_array(self).astype(numpy.bool8)