from main.gp.Covariance import Covariance
from main.kernel.GaussianKernel import GaussianKernel
from numpy.ma.core import asarray

class SquaredExponentialCovariance(Covariance):
    def __init__(self, theta=asarray([1,1])):
        """
        theta is a 1d-array which contains
        scale - is multiplied to kernel
        sigma - kernel width 
        """
        Covariance.__init__(self)
        self.gaussian=GaussianKernel(0);
        self.set_parameters(theta)
        
    def get_dim_theta(self):
        return 2
    
    def set_theta(self, theta):
        self.scale=theta[0]
        self.gaussian.width=theta[1]
        
    def get_theta(self):
        return asarray([self.scale, self.gaussian.width])
    
    def compute(self, X, Y=None):
        return self.scale*self.gaussian.kernel(X, Y)