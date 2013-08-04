from main.distribution.Gaussian import Gaussian
from main.kernel.Kernel import Kernel
from main.mcmc.samplers.MCMCSampler import MCMCSampler
from numpy import eye
from numpy.linalg import cholesky
from numpy.ma.core import shape

class Kameleon(MCMCSampler):
    """
    MCMC Hammer with oracle samples Z
    """
    def __init__(self, distribution, kernel, Z, nu2=0.1, gamma=None):
        MCMCSampler.__init__(self, distribution)
        
        self.kernel = kernel
        self.nu2 = nu2
        if gamma is not None:
            self.gamma = gamma
        else:
            self.gamma=0.5
        self.Z = Z
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "kernel="+ str(self.kernel)
        s += ", nu2="+ str(self.nu2)
        s += ", gamma="+ str(self.gamma)
        s += ", " + MCMCSampler.__str__(self)
        s += "]"
        return s
    
    def compute_constants(self, y):
        """
        Precomputes constants of the log density of the proposal distribution,
        which is Gaussian as p(x|y) ~ N(mu, R)
        where
        mu = y -a
        a = 0
        R  = gamma^2 I + M M^T
        M  = 2 [\nabla_x k(x,z_i]|_x=y
        
        Returns (mu,L_R), where L_R is lower Cholesky factor of R
        """
        assert(len(shape(y))==1)
        
        # M = 2 [\nabla_x k(x,z_i]|_x=y
        if self.Z is None:
            R = self.gamma ** 2 * eye(len(y))
        else:
            M = 2 * self.kernel.gradient(y, self.Z)
            # R = gamma^2 I + \nu^2 * M H M^T
            H = Kernel.centring_matrix(len(self.Z))
            R = self.gamma ** 2 * eye(len(y)) + self.nu2 * M.T.dot(H.dot(M))
            
        L_R = cholesky(R)
        
        return y.copy(), L_R
    
    def construct_proposal(self, y):
        """
        Returns the proposal distribution at point y given the current history
        """
        assert(len(shape(y))==1)
        
        mu, L_R = self.compute_constants(y)
        return Gaussian(mu, L_R, is_cholesky=True)
    
    def adapt(self, mcmc_chain, step_output):
        """
        Nothing for this one since it uses oracle samples
        """
        
