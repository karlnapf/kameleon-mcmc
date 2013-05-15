from numpy.ma.core import log
from numpy.matlib import rand

class MCMCSampler(object):
    def __init__(self, distribution):
        self.distribution = distribution
        self.Q = None
    
    def init(self, start):
        self.current = start
    
    def update(self, samples, ratios):
        raise NotImplementedError()
    
    def construct_proposal(self, y):
        raise NotImplementedError()
    
    def step(self):
        """
        Performs on Metropolis-Hastings step, updates internal state and returns
        
        sample, proposal, accepted, log_lik, log_ratio where
        sample - new or old sample (row-vector)
        accepted - boolean whether accepted
        log_lik - log-likelihood of returned sample
        log_ratio - log probability of acceptance
        """
        # first step
        if self.Q is None:
            self.Q = self.construct_proposal(self.current)
        
        proposal = self.Q.sample(1)
        Q_new = self.construct_proposal(proposal)
        
        log_lik_proposal = self.distribution.log_pdf(proposal)
        log_lik_current = self.distribution.log_pdf(self.current)
        log_Q_proposal = self.Q.log_pdf(proposal)
        log_Q_current = Q_new.log_pdf(self.current)
        
        log_ratio = log_lik_proposal - log_Q_proposal - log_lik_current + log_Q_current
        log_ratio = min(log(1), log_ratio)
        
        accepted = log_ratio > log(rand(1))
        
        if accepted:
            sample = proposal
            log_lik = log_lik_proposal
            self.Q = Q_new
        else:
            sample = self.current.copy()
            log_lik = log_lik_current
            
        # update state: position and proposal
        self.current = sample.copy()
            
        return sample, proposal, accepted, log_lik, log_ratio
