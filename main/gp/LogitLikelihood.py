"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from main.gp.Likelihood import Likelihood
from numpy.ma.core import log, exp

class LogitLikelihood(Likelihood):
    def __init__(self):
        pass
        
    def log_lik(self, y, f):
        return sum(-log(1+exp(-y*f)))
