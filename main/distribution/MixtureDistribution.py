from main.distribution.Discrete import Discrete
from main.distribution.Distribution import Distribution
from main.distribution.Gaussian import Gaussian
from main.tools.MatrixTools import MatrixTools
from main.tools.Visualise import Visualise
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import cholesky
from numpy.ma.core import array, zeros, log, exp, ones
from scipy.constants.constants import pi

class MixtureDistribution(Distribution):
    """
    mixing_proportion is of class Distribution->Discrete
    components is a list of Distributions
    """
    def __init__(self, dimension=2, num_components=2, components=None, mixing_proportion=None):
        Distribution.__init__(self, dimension)
        self.num_components = num_components
        if (components == None):
            self.components = [Gaussian(mu=zeros(self.dimension)) for _ in range(self.num_components)]
        else:
            assert(len(components)==self.num_components)
            self.components=components
        if (mixing_proportion == None):
            self.mixing_proportion=Discrete((1.0/num_components)*ones([num_components]))
        else:
            assert(num_components==mixing_proportion.num_objects)
            self.mixing_proportion = mixing_proportion

        
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
            jj = self.mixing_proportion.sample()
            rez[ii, :] = self.components[int(jj)].sample()
        return rez
    
if __name__ == '__main__':
    mu = array([5, 2])
    Sigma = eye(2)
    Sigma[0, 0] = 20
    R = MatrixTools.rotation_matrix(pi / 4)
    Sigma = R.dot(Sigma).dot(R.T)
    L = cholesky(Sigma)
    g1 = Gaussian(mu, L, is_cholesky=True)
    g2 = Gaussian()
    m = MixtureDistribution(dimension=2, num_components=2, components=[g1,g2])
    Z1 = g1.sample(1000)
    Z = m.sample(100)
    Visualise.visualise_distribution(g1,Z1)
    Visualise.visualise_distribution(m, Z)
    
