from main.kernel.Kernel import Kernel
from numpy.ma.core import exp
from scipy.spatial.distance import squareform, pdist, cdist

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        Kernel.__init__(self)
        
        self.width = sigma
        
    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)
        
        X - samples on right hand side
        Y - samples on left hand side, can be None in which case its replaced by X
        """
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            sq_dists = cdist(X, Y, 'sqeuclidean')
    
        K = exp(-0.5 * (sq_dists) / self.width ** 2)
        return K
    
    def gradient(self, x, Y):
        """
        Computes the gradient of the Gaussian kernel wrt. to the left argument, i.e.
        k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2), which is
        \nabla_x k(x,y)=1.0/sigma**2 k(x,y)(y-x)
        Given a set of row vectors Y, this computes the
        gradient for every pair (x,y) for y in Y.
        
        x - single sample on right hand side
        Y - samples on left hand side
        """
        k = self.kernel(x, Y)
        differences = Y - x
        G = (1.0 / self.width ** 2) * (k.T * differences)
        return G
