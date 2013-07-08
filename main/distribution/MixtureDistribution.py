"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from main.distribution.Discrete import Discrete
from main.distribution.Distribution import Distribution, Sample
from main.distribution.Gaussian import Gaussian
from numpy.lib.twodim_base import eye
from numpy.ma.core import zeros, log, exp, ones

class MixtureDistribution(Distribution):
    """
    mixing_proportion is of class Distribution->Discrete
    components is a list of Distributions
    """
    def __init__(self, dimension=2, num_components=2, components=None, mixing_proportion=None):
        Distribution.__init__(self, dimension)
        self.num_components = num_components
        if (components == None):
            self.components = [Gaussian(mu=zeros(self.dimension),Sigma=eye(self.dimension)) for _ in range(self.num_components)]
        else:
            assert(len(components)==self.num_components)
            self.components=components
        if (mixing_proportion == None):
            self.mixing_proportion=Discrete((1.0/num_components)*ones([num_components]))
        else:
            assert(num_components==mixing_proportion.num_objects)
            self.mixing_proportion = mixing_proportion

    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "components="+ str(self.components)
        s += ", mixing_proportion="+ str(self.mixing_proportion)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s
     
    def log_pdf(self, X, component_index_given=None):
        """
        If component_index_given is given, then just condition on it,
        otherwise, should compute the overall log_pdf
        """
        if component_index_given == None:
            rez = zeros([len(X)])
            for ii in range(len(X)):
                logpdfs = zeros([self.num_components])
                for jj in range(self.num_components):
                    logpdfs[jj] = self.components[jj].log_pdf([X[ii]])
                lmax = max(logpdfs)
                rez[ii] = lmax + log(sum(self.mixing_proportion.omega * exp(logpdfs - lmax)))
            return rez
        else:
            assert(component_index_given < self.num_components)
            return self.components[component_index_given].log_pdf(X)
    
    def sample(self, n=1):
        rez = zeros([n, self.dimension])
        for ii in range(n):
            which_component = self.mixing_proportion.sample().samples
            rez[ii, :] = self.components[which_component].sample().samples
            
        return SampleFromMixture(rez,which_component)
    
class SampleFromMixture(Sample):
    def __init__(self,samples,which_component):
        Sample.__init__(self,samples)
        self.which_component=which_component
        
#if __name__ == '__main__':
#    mu = array([5, 2])
#    Sigma = eye(2)
#    Sigma[0, 0] = 20
#    R = MatrixTools.rotation_matrix(pi / 4)
#    Sigma = R.dot(Sigma).dot(R.T)
#    L = cholesky(Sigma)
#    g1 = Gaussian(mu, L, is_cholesky=True)
#    g2 = Gaussian()
#    m = MixtureDistribution(dimension=2, num_components=2, components=[g1,g2])
#    Z1 = g1.sample(1000).samples
#    Z = m.sample(100).samples
#    Visualise.visualise_distribution(g1,Z1)
#    Visualise.visualise_distribution(m, Z)
#    
