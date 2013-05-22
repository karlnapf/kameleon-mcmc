from main.distribution.Banana import Banana
from main.kernel.Kernel import Kernel
from matplotlib.pyplot import imshow, show
from numpy.core.numeric import zeros
from numpy.ma.core import exp, shape, reshape
from scipy.spatial.distance import squareform, pdist, cdist

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        Kernel.__init__(self)
        
        self.width = sigma
        
    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)
        
        X - 2d array, samples on right hand side
        Y - 2d array, samples on left hand side, can be None in which case its replaced by X
        """
        
        assert(len(shape(X))==2)
        assert(len(shape(Y))==2)
        assert(shape(X)[1]==shape(Y)[1])
        
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
        
        x - single sample on right hand side (1D vector)
        Y - samples on left hand side (2D matrix)
        """
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        x_2d=reshape(x, (1, len(x)))
        k = self.kernel(x_2d, Y)
        differences = Y - x
        G = (1.0 / self.width ** 2) * (k.T * differences)
        return G

if __name__ == '__main__':
    distribution = Banana()
    Z = distribution.sample(100).samples
    Z2 = distribution.sample(100).samples
    kernel = GaussianKernel(5)
    K = kernel.kernel(Z, Z2)
    grad=kernel.gradient(zeros((1,2)), Z)
    imshow(K, interpolation="nearest")
    show()
