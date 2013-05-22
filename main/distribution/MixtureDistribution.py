from main.distribution.Discrete import Discrete
from main.distribution.Distribution import Distribution
from main.distribution.Gaussian import Gaussian
from main.tools.MatrixTools import MatrixTools
from main.tools.Visualise import Visualise
from numpy.dual import cholesky
from numpy.lib.twodim_base import eye
from numpy.ma.core import array, zeros, log, exp
from scipy.constants.constants import pi

class MixtureDistribution(Distribution):
    """
    MixingProportion is of class Distribution->Discrete
    Components is a list of Distributions
    """
    def __init__(self, MixingProportion=Discrete([0.5, 0.5]), Components=[Gaussian(array([-1, -1])), Gaussian(array([1, 1]))]):
        Distribution.__init__(self, Components[0].dimension)
        assert(MixingProportion.NumObjects == len(Components))
        self.NumComponents = MixingProportion.NumObjects
        self.MixingProportion = MixingProportion
        self.Components = Components
        
    def log_pdf(self, X, ComponentIndexGiven=None):
        """
        If ComponentIndexGiven is given, then just condition on it,
        otherwise, should compute the overall log_pdf
        """
        if ComponentIndexGiven == None:
            rez = zeros([len(X)])
            for ii in range(len(X)):
                logpdfs = zeros([self.NumComponents])
                for jj in range(self.NumComponents):
                    logpdfs[jj] = self.Components[jj].log_pdf([X[ii]])
                lmax = max(logpdfs)
                rez[ii] = lmax + log(sum(self.MixingProportion.omega * exp(logpdfs - lmax)))
            return rez
        else:
            assert(ComponentIndexGiven < self.NumComponents)
            return self.Components[ComponentIndexGiven].log_pdf(X)
    
    def sample(self, n=1):
        rez = zeros([n, self.dimension])
        for ii in range(n):
            jj = self.MixingProportion.sample()
            rez[ii, :] = self.Components[int(jj)].sample()
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
    Z1=g1.sample(1000)
    Z = m.sample(100)
    Visualise.visualise_distribution(g1,Z1)
    Visualise.visualise_distribution(m, Z)
    
    
