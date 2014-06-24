"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from abc import abstractmethod
from numpy import eye, concatenate, zeros, shape, mean, reshape, arange, exp, outer
from numpy.random import permutation,shuffle
from numpy.lib.index_tricks import fill_diagonal
from matplotlib.pyplot import imshow,show
from kameleon_mcmc.tools.HelperFunctions import HelperFunctions


class Kernel(object):
    def __init__(self):
        pass
    
    def __str__(self):
        s=self.__class__.__name__+ "=[]"
        return s
    
    @abstractmethod
    def kernel(self, X, Y=None):
        raise NotImplementedError()
    
    @abstractmethod
    def gradient(self, x, Y):
        
        # ensure this in every implementation
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        raise NotImplementedError()
    
    @staticmethod
    def centring_matrix(n):
        """
        Returns the centering matrix eye(n) - 1.0 / n
        """
        return eye(n) - 1.0 / n
    
    @staticmethod
    def center_kernel_matrix(K):
        """
        Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
        """
        n = shape(K)[0]
        H = eye(n) - 1.0 / n
        return  1.0 / n * H.dot(K.dot(H))
    
    @abstractmethod
    def show_kernel_matrix(self,X,Y=None):
        K=self.kernel(X,Y)
        imshow(K, interpolation="nearest")
        show()
    
    @abstractmethod
    def estimateMMD(self,sample1,sample2,unbiased=False):
        """
        Compute the MMD between two samples
        """
        K11 = self.kernel(sample1,sample1)
        K22 = self.kernel(sample2,sample2)
        K12 = self.kernel(sample1,sample2)
        if unbiased:
            fill_diagonal(K11,0.0)
            fill_diagonal(K22,0.0)
            n=float(shape(K11)[0])
            m=float(shape(K22)[0])
            return sum(sum(K11))/(pow(n,2)-n) + sum(sum(K22))/(pow(m,2)-m) - 2*mean(K12[:])
        else:
            return mean(K11[:])+mean(K22[:])-2*mean(K12[:])
        
    @abstractmethod
    def TwoSampleTest(self,sample1,sample2,numShuffles=1000,method='vanilla',blockSize=20):
        """
        Compute the p-value associated to the MMD between two samples
        method determines the null approximation procedure:
        ----'vanilla': standard permutation test
        ----'block': block permutation test
        ----'wild': wild bootstrap
        ----'wild-center': wild bootstrap with empirical degeneration
        """
        n1=shape(sample1)[0]
        n2=shape(sample2)[0]
        merged = concatenate( [sample1, sample2], axis=0 )
        merged_len=shape(merged)[0]
        numBlocks = merged_len/blockSize
        K=self.kernel(merged)
        mmd = mean(K[:n1,:n1])+mean(K[n1:,n1:])-2*mean(K[n1:,:n1])
        null_samples = zeros(numShuffles)
        
        if method=='vanilla':
            for i in range(numShuffles):
                pp = permutation(merged_len)
                Kpp = K[pp,:][:,pp]
                null_samples[i] = mean(Kpp[:n1,:n1])+mean(Kpp[n1:,n1:])-2*mean(Kpp[n1:,:n1])
                
        elif method=='block':
            blocks=reshape(arange(merged_len),(numBlocks,blockSize))
            for i in range(numShuffles):
                pb = permutation(numBlocks)
                pp = reshape(blocks[pb],(merged_len))
                Kpp = K[pp,:][:,pp]
                null_samples[i] = mean(Kpp[:n1,:n1])+mean(Kpp[n1:,n1:])-2*mean(Kpp[n1:,:n1])
                
        elif method=='wild' or method=='wild-center':
            if n1!=n2:
                raise ValueError("Wild bootstrap MMD available only on the same sample sizes")
            alpha = exp(-1/float(blockSize))
            coreK = K[:n1,:n1]+K[n1:,n1:]-K[n1:,:n1]-K[:n1,n1:]
            for i in range(numShuffles):
                """
                w is a draw from the Ornstein-Uhlenbeck process
                """
                w = HelperFunctions.generateOU(n=n1,alpha=alpha)
                if method=='wild-center':
                    """
                    empirical degeneration (V_{n,2} in Leucht & Neumann)
                    """
                    w = w - mean(w)
                null_samples[i]=mean(outer(w,w)*coreK)
        elif method=='wild2':
            
            alpha = exp(-1/float(blockSize))
            for i in range(numShuffles):
                wx=HelperFunctions.generateOU(n=n1,alpha=alpha)
                wx = wx - mean(wx)
                wy=HelperFunctions.generateOU(n=n2,alpha=alpha)
                wy = wy - mean(wy)
                null_samples[i]=mean(outer(wx,wx)*K[:n1,:n1])+mean(outer(wy,wy)*K[n1:,n1:])-2*mean(outer(wx,wy)*K[:n1,n1:])
        else:
            raise ValueError("Unknown null approximation method")
        return sum(mmd<null_samples)/float(numShuffles)
