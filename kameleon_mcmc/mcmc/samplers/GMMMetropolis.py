from modshogun import GMM, RealFeatures
from numpy import zeros
from numpy.ma.extras import unique
from numpy.random import randint

from kameleon_mcmc.distribution.Discrete import Discrete
from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.distribution.MixtureDistribution import MixtureDistribution
from kameleon_mcmc.mcmc.samplers.StandardMetropolis import StandardMetropolis


class GMMMetropolis(StandardMetropolis):
    '''
    Runs StandardMetropolis for a number of iterations, performs a couple of
    EM instances to fit a Gaussian Mixture Model which is subsequently used
    as a static proposal distribution
    '''
    
    def __init__(self, distribution, num_components, num_sample_discard=1000,
                 num_samples_gmm=1000, num_samples_when_to_switch=40000, num_runs_em=1):
        StandardMetropolis.__init__(self, distribution)
        
        self.num_components = num_components
        self.num_sample_discard = num_sample_discard
        self.num_samples_gmm = num_samples_gmm
        self.num_samples_when_to_switch = num_samples_when_to_switch
        self.num_runs_em = num_runs_em
        
        # start with empty proposal, is changed to something in adapt method
        self.proposal = None
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "num_components=" + str(self.num_components)
        s += ", num_sample_discard=" + str(self.num_sample_discard)
        s += ", num_samples_gmm=" + str(self.num_samples_gmm)
        s += ", num_runs_em=" + str(self.num_runs_em)
        s += ", " + StandardMetropolis.__str__(self)
        s += "]"
        return s
    
    def adapt(self, mcmc_chain, step_output):
        # only learn the proposal once, at a pre-specified iteration
        if mcmc_chain.iteration == self.num_samples_when_to_switch:
            iter_no = mcmc_chain.iteration
            inds = randint(iter_no - self.num_sample_discard, size=self.num_samples_gmm) + self.num_sample_discard
            unique_inds = unique(inds)
            self.proposal = self.fit_gmm(mcmc_chain.samples[unique_inds])
            
            #idx_left = self.num_sample_discard
            #idx_right = self.num_sample_discard + self.num_samples_gmm
            #samples = mcmc_chain.samples[idx_left:idx_right]
            #self.proposal = self.fit_gmm(samples)
    
    def construct_proposal(self, y):
        # fixed proposal exists from a certain iteration, return std MH otherwise
        # was created in adapt method
        if self.proposal is not None:
            return self.proposal
        else:
            return StandardMetropolis.construct_proposal(self, y)
    
    def fit_gmm(self, samples):
        """
        Runs a couple of em instances on random starting points and returns
        internal GMM representation of best instance
        """
        features = RealFeatures(samples.T)
        
        gmms = []
        log_likelihoods = zeros(self.num_runs_em)
        for i in range(self.num_runs_em):
            # set up Shogun's GMM class and run em (corresponds to random
            # initialisation)
            gmm = GMM(self.num_components)
            gmm.set_features(features)
            log_likelihoods[i] = gmm.train_em()
            gmms.append(gmm)
            
        
        max_idx = log_likelihoods.argmax()

        # construct Gaussian mixture components in internal representation
        components = []
        for i in range(self.num_components):
            mu = gmms[max_idx].get_nth_mean(i)
            Sigma = gmms[max_idx].get_nth_cov(i)
            components.append(Gaussian(mu, Sigma))
            
        # construct a Gaussian mixture model based on the best EM run
        pie = gmms[max_idx].get_coef()
        proposal = MixtureDistribution(components[0].dimension,
                                     self.num_components, components,
                                     Discrete(pie))
        
        return proposal
