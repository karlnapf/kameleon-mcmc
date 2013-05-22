from abc import abstractmethod
from main.distribution.Distribution import Sample
from numpy.ma.core import log, shape, reshape
from numpy.random import rand

class MCMCSampler(object):
    def __init__(self, distribution):
        self.distribution = distribution
        self.Q = None
    
    def init(self, start):
        assert(len(shape(start)) == 1)
        
        self.current_sample_object = Sample(start)
        start_2d = reshape(start, (1, len(start)))
        self.log_lik_current = self.distribution.log_pdf(start_2d)
    
    @abstractmethod
    def adapt(self, mcmc_chain, step_output):
        raise NotImplementedError()
    
    @abstractmethod
    def construct_proposal(self, y):
        """
        parameters:
        y - 1D array with a current_sample_object point
        """
        
        # ensure this in every implementation
        assert(len(shape(y)) == 1)
        raise NotImplementedError()
    
    def step(self):
        """
        Performs on Metropolis-Hastings step, updates internal state and returns
        
        sample_object, proposal_2d, accepted, log_lik, log_ratio where
        sample_object - new or old sample_object (row-vector)
        accepted - boolean whether accepted
        log_lik - log-likelihood of returned sample_object
        log_ratio - log probability of acceptance
        """
        # create proposal around current_sample_object point in first step only
        dim = self.distribution.dimension
        if self.Q is None:
            current_1d=reshape(self.current_sample_object.samples, (dim,))
            self.Q = self.construct_proposal(current_1d)
        
        # propose sample_object and construct new Q centred at proposal_2d
        proposal_object=self.Q.sample(1)
        proposal_2d = proposal_object.samples
        proposal_1d = reshape(proposal_2d, (dim,))
        Q_new = self.construct_proposal(proposal_1d)
        

        # 2d view for current_sample_object point
        current_2d = reshape(self.current_sample_object.samples, (1, dim))
        
        # evaluate both Q
        log_Q_proposal_given_current = self.Q.log_pdf(proposal_2d)
        log_Q_current_given_proposal = Q_new.log_pdf(current_2d)
        
        log_lik_proposal = self.distribution.log_pdf(proposal_2d)
        log_ratio = log_lik_proposal - self.log_lik_current \
                    + log_Q_current_given_proposal - log_Q_proposal_given_current
        log_ratio = min(log(1), log_ratio)
        
        accepted = log_ratio > log(rand(1))
        
        if accepted:
            sample_object = proposal_object
            self.log_lik_current = log_lik_proposal
            self.Q = Q_new
        else:
            sample_object = self.current_sample_object
            
        # adapt state: position and proposal_2d
        self.current_sample_object = sample_object
            
        return StepOutput(sample_object, proposal_object, accepted, self.log_lik_current, log_ratio)


class StepOutput(object):
    def __init__(self, sample_object, proposal_object, accepted, log_lik, log_ratio):
        self.sample = sample_object
        self.proposal_object = proposal_object
        self.accepted = accepted
        self.log_lik = log_lik
        self.log_ratio = log_ratio
        
