from main.gp.Likelihood import Likelihood
from numpy.ma.core import log, exp

class LogitLikelihood(Likelihood):
    def __init__(self):
        pass
        
    def log_lik(self, y, f):
        return sum(-log(1+exp(-y*f)))
