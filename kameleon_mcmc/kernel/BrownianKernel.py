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


from matplotlib.pyplot import show, imshow
from numpy import shape, reshape
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist

from kameleon_mcmc.distribution.Banana import Banana
from kameleon_mcmc.kernel.Kernel import Kernel
from kameleon_mcmc.tools.GenericTests import GenericTests


class BrownianKernel(Kernel):
    def __init__(self, alpha=1.0):
        Kernel.__init__(self)
        GenericTests.check_type(alpha,'alpha',float)
        
        self.alpha = alpha
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "alpha="+ str(self.alpha)
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        
        GenericTests.check_type(X,'X',np.ndarray,2)
        # if X=Y, use more efficient pdist call which exploits symmetry
        normX=reshape(np.linalg.norm(X,axis=1),(len(X),1))
        if Y is None:
            dists = squareform(pdist(X, 'euclidean'))
            normY=normX.T
        else:
            GenericTests.check_type(Y,'Y',np.ndarray,2)
            assert(shape(X)[1]==shape(Y)[1])
            normY=reshape(np.linalg.norm(Y,axis=1),(1,len(Y)))
            dists = cdist(X, Y, 'euclidean')
        K=0.5*(normX**self.alpha+normY**self.alpha-dists**self.alpha)
        return K
    
    def gradient(self, x, Y):
        raise NotImplementedError()
    
if __name__ == '__main__':
    distribution = Banana()
    Z = distribution.sample(50).samples
    Z2 = distribution.sample(50).samples
    kernel = BrownianKernel()
    K = kernel.kernel(Z, Z2)
    imshow(K, interpolation="nearest")
    show()
