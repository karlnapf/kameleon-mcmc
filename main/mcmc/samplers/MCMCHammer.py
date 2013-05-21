from main.distribution.Banana import Banana
from main.distribution.Gaussian import Gaussian
from main.distribution.Ring import Ring
from main.kernel.GaussianKernel import GaussianKernel
from main.kernel.Kernel import Kernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.samplers.MCMCSampler import MCMCSampler
from main.tools.Visualise import Visualise
from numpy import eye
from numpy.core.function_base import linspace
from numpy.core.numeric import array
from numpy.dual import cholesky
from numpy.ma.core import shape

class MCMCHammer(MCMCSampler):
    """
    MCMC Hammer with oracle samples Z
    """
    def __init__(self, distribution, kernel, Z, eta=0.1, gamma=0.1):
        MCMCSampler.__init__(self, distribution)
        
        self.kernel = kernel
        self.eta = eta
        self.gamma = gamma
        self.Z = Z
    
    def compute_constants(self, y):
        """
        Precomputes constants of the log density of the proposal distribution,
        which is Gaussian as p(x|y) ~ N(mu, R)
        where
        mu = y -a
        a = 0
        R  = gamma^2 I + M M^T
        M  = 2\eta [\nabla_x k(x,z_i]|_x=y
        
        Returns (mu,L_R), where L_R is lower Cholesky factor of R
        """
        dim = shape(y)[1]
            
        # we think that a=0 for every kernel
        mu = y
        
        # M = 2 [\nabla_x k(x,z_i]|_x=y
        M = 2 * self.kernel.gradient(y, self.Z)
        
        # R = gamma^2 I + \eta^2 * M H M^T
        H = Kernel.centring_matrix(len(self.Z))
        R = self.gamma ** 2 * eye(dim) + self.eta ** 2 * M.T.dot(H.dot(M))
        L_R = cholesky(R)
        
        return mu, L_R
    
    def construct_proposal(self, y):
        """
        Returns the proposal distribution at point y given the current history
        """
        mu, L_R = self.compute_constants(y)
        return Gaussian(mu, L_R, is_cholesky=True)
    
    def update(self, samples, ratios):
        """
        Nothing for this one since it uses oracle samples
        """
        
if __name__ == '__main__':
    distribution = Ring()
    Z = distribution.sample(1000)
    kernel = GaussianKernel(sigma=1)
    mcmc_sampler = MCMCHammer(distribution, kernel, Z)
    
    start = array([[-2, -2]])
    mcmc_params = MCMCParams(start=start, num_iterations=10000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(ProgressOutput())
    Xs = linspace(-5, 5, 50)
    Ys = linspace(-5, 5, 50)
    chain.append_mcmc_output(PlottingOutput(distribution, plot_from=1))
    chain.run()
    
    Visualise.visualise_distribution(distribution, chain.samples)
