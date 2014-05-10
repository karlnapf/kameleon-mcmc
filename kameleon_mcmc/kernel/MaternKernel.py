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


from kameleon_mcmc.kernel.Kernel import Kernel
from numpy.ma.core import exp, shape, sqrt, reshape
from scipy.spatial.distance import squareform, pdist, cdist
from matplotlib.pyplot import show, imshow
from kameleon_mcmc.distribution.Banana import Banana
import numpy
from kameleon_mcmc.tools.GenericTests import GenericTests


class MaternKernel(Kernel):
    def __init__(self, rho, nu=1.5, sigma=1.0):
        Kernel.__init__(self)
        #GenericTests.check_type(rho,'rho',float)
        GenericTests.check_type(nu,'nu',float)
        GenericTests.check_type(sigma,'sigma',float)
        
        self.rho = rho
        self.nu = nu
        self.sigma = sigma
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "rho="+ str(self.rho)
        s += "nu="+ str(self.nu)
        s += "sigma="+ str(self.sigma)
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        
        GenericTests.check_type(X,'X',numpy.ndarray,2)
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            dists = squareform(pdist(X, 'euclidean'))
        else:
            GenericTests.check_type(Y,'Y',numpy.ndarray,2)
            assert(shape(X)[1]==shape(Y)[1])
            dists = cdist(X, Y, 'euclidean')
        if self.nu==0.5:
            #for nu=1/2, Matern class corresponds to Ornstein-Uhlenbeck Process
            K = (self.sigma**2.) * exp( -dists / self.rho )                 
        elif self.nu==1.5:
            K = (self.sigma**2.) * (1+ sqrt(3.)*dists / self.rho) * exp( -sqrt(3.)*dists / self.rho )
        elif self.nu==2.5:
            K = (self.sigma**2.) * (1+ sqrt(5.)*dists / self.rho + 5.0*(dists**2.) / (3.0*self.rho**2.) ) * exp( -sqrt(5.)*dists / self.rho )
        else:
            raise NotImplementedError()
        return K
    
    def gradient(self, x, Y):
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        if self.nu==1.5 or self.nu==2.5:
            x_2d=reshape(x, (1, len(x)))
            lower_order_rho = self.rho * sqrt(2*(self.nu-1)) / sqrt(2*self.nu)
            lower_order_kernel = MaternKernel(lower_order_rho,self.nu-1,self.sigma)
            k = lower_order_kernel.kernel(x_2d, Y)
            differences = Y - x
            G = ( 1.0 / lower_order_rho ** 2 ) * (k.T * differences)
            return G
        else:
            raise NotImplementedError()
    
if __name__ == '__main__':
    distribution = Banana()
    Z = distribution.sample(50).samples
    Z2 = distribution.sample(50).samples
    kernel = MaternKernel(5.0, nu=1.5, sigma=2.0)
    K = kernel.kernel(Z, Z2)
    imshow(K, interpolation="nearest")
    #G = kernel.gradient(Z[0],Z2)
    #print G
    show()
