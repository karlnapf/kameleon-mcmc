from main.distribution.Discrete import Discrete
from main.distribution.Gaussian import Gaussian
from main.distribution.MixtureDistribution import MixtureDistribution
from main.distribution.Ring import Ring
from main.kernel.GaussianKernel import GaussianKernel
from main.kernel.Kernel import Kernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.samplers.MCMCHammer import MCMCHammer
from main.tools.Visualise import Visualise
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import svd
from numpy.ma.core import shape, outer, array

class KameleonPCA(MCMCHammer):
    '''
    PCA version of the Kameleon MCMC sampler
    performs eigendecomposition of the centred kernel matrix HKH
    to inform proposals
    '''
    def __init__(self, distribution, kernel, Z, eta=0.1, gamma=0.1, num_eigen=10):
        MCMCHammer.__init__(self, distribution, kernel, Z, eta, gamma)
        self.num_eigen = num_eigen
        if Z is None:
            self.Kc=None
            self.eigvalues=None
            self.eigvectors=None
        else:
            K=self.kernel.kernel(Z)
            H=Kernel.centring_matrix(len(self.Z))
            self.Kc=H.dot(K.dot(H))
            u, s, _ = svd(self.Kc)
            self.eigvalues = s[0:self.num_eigen]
            self.eigvectors = u[:, 0:self.num_eigen]
        
        
    def construct_proposal(self, y):
        """
        proposal is a mixture of normals,
        centred at y and with covariance gamma^2 I + eta^2 MHaa'HM',
        where a are the eigenvectors of centred kernel matrix Kc=HKH
        """
        assert(len(shape(y)) == 1)
        m = MixtureDistribution(self.distribution.dimension, self.num_eigen)
        m.mixing_proportion = Discrete((self.eigvalues + 1) / (sum(self.eigvalues) + self.num_eigen))
        # print "current mixing proportion: ", m.mixing_proportion.omega
        M = 2 * self.kernel.gradient(y, self.Z)
        H = Kernel.centring_matrix(len(self.Z))
        
        for ii in range(self.num_eigen):
            Sigma = self.gamma ** 2 * eye(len(y)) + \
            self.eta ** 2 * (M.T).dot(H.dot(outer(self.eigvectors[:, ii], self.eigvectors[:, ii]).dot(H.dot(M))))
            m.components[ii] = Gaussian(y, Sigma)
        return m
    
if __name__ == '__main__':
    distribution = Ring()
    Z = distribution.sample(200).samples
    kernel = GaussianKernel(sigma=1)
    mcmc_sampler = KameleonPCA(distribution, kernel, Z, eta=.5, num_eigen=2)
    
    start = array([-2, -2])
    mcmc_params = MCMCParams(start=start, num_iterations=5000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(ProgressOutput())
    chain.append_mcmc_output(PlottingOutput(distribution, plot_from=1))
    chain.run()
    
    Visualise.visualise_distribution(distribution, chain.samples)
