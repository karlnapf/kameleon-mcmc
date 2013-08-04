"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from main.gp.likelihood.Likelihood import Likelihood
from numpy.ma.core import log, exp, asarray

class LogitLikelihood(Likelihood):
    """
    Likelihood for binary logit regression, taken drom GPML toolbox under GPL
    """
    def __init__(self):
        pass
        
    def log_lik_vector(self, y, f):
        s = -y * f
        ps = asarray([min(x, 0) for x in s])
        lp = -(ps + log(exp(-ps) + exp(s - ps)))
        return lp
    
    def log_lik_vector_multiple(self, y, F):
        S = -y * F
        PS = asarray([asarray([min(x, 0) for x in s]) for s in S])
        LP = -(PS + log(exp(-PS) + exp(S - PS)))
        return LP

    def log_lik_grad_vector(self, y, f):
        s = asarray([min(x, 0) for x in f])
        p = exp(s) * (exp(s) + exp(s - f));
        dlp = (y + 1) / 2 - p
        return dlp
    
    def log_lik_hessian_vector(self, y, f):
        s = asarray([min(x, 0) for x in f])
        d2lp = -exp(2 * s - f) / (exp(s) + exp(s - f)) ** 2
        return d2lp

    def gen_num_hyperparameters(self):
        return 0
    
    def get_hyperparameters(self):
        return None
    
    def set_hyperparameters(self, theta):
        pass