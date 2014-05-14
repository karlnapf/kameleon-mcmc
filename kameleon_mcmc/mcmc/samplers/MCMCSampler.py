from abc import abstractmethod
from numpy.ma.core import log, shape, reshape
from numpy.random import rand

from kameleon_mcmc.distribution.Distribution import Sample


class MCMCSampler(object):
    def __init__(self, distribution):
        self.distribution = distribution
        self.Q = None
        self.is_symmetric = False
        
    def init(self, start):
        assert(len(shape(start)) == 1)
        
        self.current_sample_object = Sample(start)
        start_2d = reshape(start, (1, len(start)))
        self.log_lik_current = self.distribution.log_pdf(start_2d)
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "distribution=" + str(self.distribution)
        s += ", is_symmetric=" + str(self.is_symmetric)
        s += "]"
        return s
    
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
            current_1d = reshape(self.current_sample_object.samples, (dim,))
            self.Q = self.construct_proposal(current_1d)
        
        # propose sample_object and construct new Q centred at proposal_2d
        proposal_object = self.Q.sample(1)
        proposal_2d = proposal_object.samples
        proposal_1d = reshape(proposal_2d, (dim,))
        Q_new = self.construct_proposal(proposal_1d)
        
        # 2d view for current_sample_object point
        current_2d = reshape(self.current_sample_object.samples, (1, dim))
        
        # First find out whether this sampler is gibbs (which has a full target)
        # or a MH (otherwise). This isn't really necessary since log-pdf of Gibbs
        # could just return 1 always, but it avoids such unnecessary method calls
        # (at the cost of code readability)
        try:
            # if this is possible, it means this is a gibbs sampler, so accept
            full_target = self.distribution.full_target
            self.log_lik_current = full_target.log_pdf(self.distribution.get_current_state_array())
            accepted = True
            log_ratio = log(1)
        except AttributeError:
            # do normal MH-step, compute acceptance ratio
        
            # evaluate both Q
            if not self.is_symmetric:
                log_Q_proposal_given_current = self.Q.log_pdf(proposal_2d)
                log_Q_current_given_proposal = Q_new.log_pdf(current_2d)
            else:
                log_Q_proposal_given_current = 0
                log_Q_current_given_proposal = 0
                
            log_lik_proposal = self.distribution.log_pdf(proposal_2d)
            
            log_ratio = log_lik_proposal - self.log_lik_current \
                        + log_Q_current_given_proposal - log_Q_proposal_given_current
            
            log_ratio = min(log(1), log_ratio)
        
            accepted = log_ratio > log(rand(1))
            
            if accepted:
                self.log_lik_current = log_lik_proposal
        
        if accepted:
            sample_object = proposal_object
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
        
