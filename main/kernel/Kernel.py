from numpy.lib.twodim_base import eye
from numpy.ma.core import shape

class Kernel(object):
    def __init__(self):
        pass
    
    def kernel(self, X, Y=None):
        raise NotImplementedError()
    
    def gradient(self, x, Y):
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
