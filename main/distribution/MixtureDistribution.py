from main.distribution.Discrete import Discrete
from main.distribution.Distribution import Distribution
from main.distribution.Gaussian import Gaussian
from main.tools.MatrixTools import MatrixTools
from main.tools.Visualise import Visualise
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import cholesky
from numpy.ma.core import array, zeros, log, exp
from scipy.constants.constants import pi

class MixtureDistribution(Distribution):
    """
    mixing_proportion is of class Distribution->Discrete
    components is a list of Distributions
    """
    def __init__(self, mixing_proportion=Discrete([0.5, 0.5]), components=[Gaussian(array([-1, -1])), Gaussian(array([1, 1]))]):
        Distribution.__init__(self, components[0].dimension)
        
        assert(mixing_proportion.num_objects == len(components))
        
        self.num_components = mixing_proportion.num_objects
        self.mixing_proportion = mixing_proportion
        self.components = components
        
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
    m = MixtureDistribution(Discrete([0.7, 0.3]), [g1, g2])
    Z1 = g1.sample(1000)
    Z = m.sample(100)
    Visualise.visualise_distribution(g1,Z1)
    Visualise.visualise_distribution(m, Z)
    
