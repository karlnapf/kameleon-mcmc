from main.distribution.Distribution import Distribution
from numpy.ma.core import cumsum, zeros
from scipy.stats.distributions import rand


class Discrete(Distribution):
    def __init__(self,omega,support=None):
        Distribution.__init__(self, dimension=None)
        assert(sum(omega)==1)
        if support==None:
            support=range(len(omega))
        else:
            assert(len(omega)==len(support))
        self.NumObjects=len(omega)
        self.omega=omega
        self.cdf=cumsum(omega)
        self.support=support
        
    def sample(self,n=1):
        u=rand(n)
        rez=zeros([n])
        for ii in range(0,n):
            jj=0
            while u[ii]>self.cdf[jj]:
                jj+=1
            rez[ii]=self.support[jj]
        return rez
    
    def log_pdf(self,X):
        return None
    
    
if __name__ == '__main__':
    d=Discrete([0.65,0.1,0.25])
    X=d.sample(50)
    print X