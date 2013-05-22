from abc import abstractmethod
from numpy.ma.core import log, shape, reshape
from numpy.random import rand

class MCMCSampler(object):
    def __init__(self, distribution):
        self.distribution = distribution
        self.Q = None
    
    def init(self, start):
        assert(len(shape(start))==1)
        
        self.current = start
        start_2d=reshape(start, (1, len(start)))
        self.log_lik_current = self.distribution.log_pdf(start_2d)
    
    @abstractmethod
    def adapt(self, mcmc_chain):
        raise NotImplementedError()
    
    @abstractmethod
    def construct_proposal(self, y):
        """
        parameters:
        y - 1D array with a current point
        """
        
        # ensure this in every implementation
        assert(len(shape(y))==1)
        raise NotImplementedError()
    
    def step(self):
        """
        Performs on Metropolis-Hastings step, updates internal state and returns
        
        sample, proposal_2d, accepted, log_lik, log_ratio where
        sample - new or old sample (row-vector)
        accepted - boolean whether accepted
        log_lik - log-likelihood of returned sample
        log_ratio - log probability of acceptance
        """
        # create proposal_2d around current point in first step only
        if self.Q is None:
            self.Q = self.construct_proposal(self.current)
        
        # propose sample and construct new Q centred at proposal_2d
        dim=self.distribution.dimension
        proposal_2d = self.Q.sample(1)
        proposal_1d=reshape(proposal_2d, (dim,))
        Q_new = self.construct_proposal(proposal_1d)
        

        # 2d view for current point
        current_2d=reshape(self.current, (1, dim))
        
        # evaluate both Q
        log_Q_proposal_given_current = self.Q.log_pdf(proposal_2d)
        log_Q_current_given_proposal = Q_new.log_pdf(current_2d)
        
        log_lik_proposal = self.distribution.log_pdf(proposal_2d)
        log_ratio = log_lik_proposal - self.log_lik_current \
                    + log_Q_current_given_proposal - log_Q_proposal_given_current
        log_ratio = min(log(1), log_ratio)
        
        accepted = log_ratio > log(rand(1))
        
        if accepted:
            sample = proposal_2d
            self.log_lik_current = log_lik_proposal
            self.Q = Q_new
        else:
            sample = self.current.copy()
            
        # adapt state: position and proposal_2d
        self.current = sample.copy()
            
        return sample, proposal_2d, accepted, self.log_lik_current, log_ratio
