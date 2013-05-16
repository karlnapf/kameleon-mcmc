from abc import abstractmethod
class Output(object):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, mcmc_params, proposal, samples, log_liks, Q):
        raise NotImplementedError()
    
    @abstractmethod
    def prepare(self):
        raise NotImplementedError()
    
