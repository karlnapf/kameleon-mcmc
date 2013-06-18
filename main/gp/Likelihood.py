from abc import abstractmethod

class Likelihood(object):
    def __init__(self):
        pass
        
    @abstractmethod
    def log_lik(self, y, f):
        raise NotImplementedError()
