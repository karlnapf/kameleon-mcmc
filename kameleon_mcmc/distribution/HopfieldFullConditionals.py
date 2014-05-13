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

from numpy import hstack, arange, exp, inner
from numpy.random import rand


from kameleon_mcmc.distribution.FullConditionals import FullConditionals
from kameleon_mcmc.distribution.Hopfield import Hopfield


class HopfieldFullConditionals(FullConditionals):
    """
    Implements the full conditional distributions for Hopfield network
    """
    def __init__(self, full_hopfield, current_state, schedule, index_block=None):
        if not isinstance(full_hopfield, Hopfield):
            raise TypeError("Given full Hopfield is not a Hopfield")
        
        FullConditionals.__init__(self, current_state, schedule, index_block)
        
        self.full_hopfield = full_hopfield
        
        
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "full_hopfield=" + str(self.full_hopfield)
        s += ", " + FullConditionals.__str__(self)
        s += "]"
        return s
    
    def sample_conditional(self, index, current):
        if index < 0 or index >= self.dimension:
            raise ValueError("Conditional index out of bounds")
        
        # conditioning indices: all indices but the current
        cond_inds = hstack((arange(0, index), arange(index + 1, self.dimension)))
        
        cond_prob = 1.0 / ( 1+exp(-self.full_hopfield.bias[index]- \
                       2*inner( self.full_hopfield.W[index,cond_inds],current[cond_inds] ) ) )
        return rand(1,)<cond_prob
