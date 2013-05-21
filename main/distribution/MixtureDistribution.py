from main.distribution.Discrete import Discrete
from main.distribution.Distribution import Distribution
from main.distribution.Gaussian import Gaussian
from main.tools.Visualise import Visualise
from numpy.ma.core import array, zeros, log, exp

class MixtureDistribution(Distribution):
    """
    MixingProportion is of class Distribution->Discrete
    Components is a list of Distributions
    """
    def __init__(self, MixingProportion=Discrete([0.5,0.5]), Components=[Gaussian(array([-1, -1])),Gaussian(array([1, 1]))]):
        Distribution.__init__(self, Components[0].dimension)
        assert(MixingProportion.NumObjects==len(Components))
        self.NumComponents=MixingProportion.NumObjects
        self.MixingProportion=MixingProportion
        self.Components=Components
        
    def log_pdf(self,X,ComponentIndexGiven=None):
        """
        If ComponentIndexGiven is given, then just condition on it,
        otherwise, should compute the overall log_pdf
        """
        if ComponentIndexGiven==None:
            rez=zeros([len(X)])
            for ii in range(len(X)):
                logpdfs=zeros([self.NumComponents])
                for jj in range(self.NumComponents):
                    logpdfs[jj]=self.Components[jj].log_pdf([X[ii]])
                lmax=max(logpdfs)
                rez[ii]=lmax+log(sum(self.MixingProportion.omega*exp(logpdfs-lmax)))
            return rez
        else:
            assert(ComponentIndexGiven<self.NumComponents)
            return self.Components[ComponentIndexGiven].log_pdf(X)
    
    def sample(self,n=1):
        rez=zeros([n,self.dimension])
        for ii in range(n):
            jj=self.MixingProportion.sample()
            rez[ii,:]=self.Components[int(jj)].sample()
        return rez
    
if __name__ == '__main__':
    m=MixtureDistribution()
    Z=m.sample(1)
    Visualise.visualise_distribution(m,Z)
    