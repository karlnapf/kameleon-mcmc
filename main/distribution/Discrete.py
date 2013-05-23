from main.distribution.Distribution import Distribution, Sample
from numpy.ma.core import cumsum, zeros
from scipy.stats.distributions import rand
import numpy

class Discrete(Distribution):
    def __init__(self, omega, support=None):
        Distribution.__init__(self, dimension=None)
        assert(abs(sum(omega)-1)<1e-6)
        if support == None:
            support = range(len(omega))
        else:
            assert(len(omega) == len(support))
        self.num_objects = len(omega)
        self.omega = omega
        self.cdf = cumsum(omega)
        self.support = support
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "num_objects="+ str(self.num_objects)
        s += ", omega="+ str(self.omega)
        s += ", cdf="+ str(self.cdf)
        s += ", support="+ str(self.support)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s
    
    def sample(self, n=1):
        u = rand(n)
        rez = zeros([n])
        for ii in range(0, n):
            jj = 0
            while u[ii] > self.cdf[jj]:
                jj += 1
            rez[ii] = self.support[jj]
        return Sample(rez.astype(numpy.int32))
    
    def log_pdf(self,X):
        return None

#if __name__ == '__main__':
#    d = Discrete([0.65, 0.1, 0.25])
#    X = d.sample(50).samples
#    print X
