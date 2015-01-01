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



import numpy as np
from numpy import infty
from kameleon_mcmc.kernel.Kernel import Kernel
from kameleon_mcmc.tools.GenericTests import GenericTests



class AutoRegressiveKernel(Kernel):
    def __init__(self, p=1, alpha=0.5, sigma=1.0):
        Kernel.__init__(self)       
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "p="+ str(self.p)
        s += "alpha="+ str(self.alpha)
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        
        GenericTests.check_type(X,'X',np.ndarray,2)
        if Y is None:
            Y=X
               
        nX=np.shape(X)[0]
        nY=np.shape(Y)[0]
        K=np.zeros((nX,nY))
        ii=0        
        for x in X:
            jj=0
            for y in Y:
                Ax,Bx=self.formVARmatrices(x)
                degx=np.shape(Bx)[1]
                Ay,By=self.formVARmatrices(y)
                degy=np.shape(By)[1]
                deltaMat = np.diag(np.concatenate((0.5*np.ones(degx)/degx,0.5*np.ones(degy)/degy)))
                A=np.concatenate((Ax,Ay),axis=1)
                B=np.concatenate((Bx,By),axis=1)
                Adel = A.dot(deltaMat)
                AdelAT = Adel.dot(A.T) 
                foo=np.linalg.solve(AdelAT+np.eye(self.p),Adel)
                precomputedMat=deltaMat-(deltaMat.dot(A.T)).dot(foo) 
                _,first_term = np.linalg.slogdet(AdelAT+np.eye(self.p))
                second_term = (B.dot(precomputedMat)).dot(B.T)+1.
                K[ii,jj]+= -(1-self.alpha)*first_term-self.alpha*second_term
                jj+=1
            ii+=1
        return np.exp(-0.5*K/(self.sigma**2.))
    
    
    def formVARmatrices(self,x):
        lenx=self.TimeSeriesLength(x)
        Ax=np.zeros((self.p,lenx-self.p))
        for ii in range(self.p):
            Ax[ii]=x[ii:(lenx-self.p+ii)]
        Bx=np.reshape(x[self.p:lenx],(1,lenx-self.p))
        return Ax,Bx
    
    def TimeSeriesLength(self,x):
        appended=np.flatnonzero(x==-infty)
        if appended.size:
            return appended[0]
        else:
            return len(x)
            
    
    def gradient(self, x, Y):
        raise NotImplementedError()
    
if __name__ == '__main__':
    kernel = AutoRegressiveKernel(p=3)
    nX=20
    nY=40
    maxlen=50
    X=np.random.randn(nX,maxlen)
    terminateX=np.random.randint(15,maxlen,nX)
    for ii in range(nX):
        X[ii,terminateX[ii]:]=-infty        
    Y=np.random.randn(nY,maxlen)
    terminateY=np.random.randint(15,maxlen,nY)
    for jj in range(nY):
        Y[jj,terminateY[jj]:]=-infty
    kernel.show_kernel_matrix(X)
    
