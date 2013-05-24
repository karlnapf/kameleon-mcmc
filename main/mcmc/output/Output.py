from abc import abstractmethod
class Output(object):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, mcmc_chain, step_output):
        raise NotImplementedError()
    
    @abstractmethod
    def prepare(self):
        raise NotImplementedError()
    
