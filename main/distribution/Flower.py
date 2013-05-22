from main.distribution.Distribution import Distribution
from main.distribution.Gaussian import Gaussian
from main.tools.Visualise import Visualise
from numpy.core.numeric import array, zeros
from numpy.core.shape_base import hstack
from numpy.linalg import norm
from numpy.ma.core import sqrt, cos, sin, arctan2, arange, shape
from numpy.random import rand, randn
from scipy.constants.constants import pi

class Flower(Distribution):
    def __init__(self, amplitude=1, frequency=7, variance=0.05, radius=3.5, dimension=2):
        Distribution.__init__(self, dimension)
        
        self.amplitude = amplitude
        self.frequency = frequency
        self.variance = variance
        self.radius = radius
        
        assert(dimension >= 2)
        
    def sample(self, n=1):
        # sample angles
        theta = rand(n, 1) * 2 * pi
        
        # sample radius
        radius_sample = randn(n, 1) * sqrt(self.variance) + self.radius + \
            self.amplitude * cos(self.frequency * theta)
        
        # sample points
        X = hstack((cos(theta) * radius_sample, sin(theta) * radius_sample)) 
        
        # add noise
        if self.dimension > 2:
            X = hstack((X, randn(n, self.dimension - 2)))
    
        return X
    
    def log_pdf(self, X):
        assert(len(shape(X))==2)
        assert(shape(X)[1]==self.dimension)
        
        # compute all norms
        norms = array([norm(x) for x in X])
        
        # compute angles (second component first first)
        angles = arctan2(X[:, 1], X[:, 0])
        
        # gaussian parameters
        mu = self.radius + self.amplitude * cos(self.frequency * angles)
        
        log_pdf = zeros(len(X))
        gaussian = Gaussian(array([mu[0]]), array([[self.variance]]))
        for i in range(len(X)):
            gaussian.mu = mu[i]
            log_pdf[i] = gaussian.log_pdf(array([[norms[i]]]))
        
        return log_pdf
    
    def emp_quantiles(self, X, quantiles=arange(0.1, 1, 0.1)):
        norms = array([norm(x) for x in X])
        angles = arctan2(X[:, 1], X[:, 0])
        if self.amplitude == 0:
            gaussian = Gaussian(array([self.radius]), array([[self.variance]]))
            return gaussian.emp_quantiles(array([norms]).T, quantiles)
        else:
            mu = self.radius + self.amplitude * cos(self.frequency * angles)
            overall = zeros([len(X), len(quantiles)])
            gaussian = Gaussian(array([mu[0]]), array([[self.variance]]))
            for i in range(len(X)):
                gaussian.mu = mu[i]
                overall[i, :] = gaussian.emp_quantiles(array([[norms[i]]]), quantiles)
            return sum(overall) / len(X)
    
if __name__ == '__main__':
    flower_instance = Flower()
    X = flower_instance.sample(1000)
    print flower_instance.emp_quantiles(X)
    Visualise.visualise_distribution(flower_instance)
