"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

import numpy as np

from kameleon_mcmc.distribution.Distribution import Distribution, Sample


class Discrete(Distribution):
    def __init__(self, omega, support=None):
        Distribution.__init__(self, dimension=None)
        assert(abs(sum(omega) - 1) < 1e-6)
        if support == None:
            support = range(len(omega))
        else:
            assert(len(omega) == len(support))
        self.num_objects = len(omega)
        self.omega = omega
        self.cdf = np.cumsum(omega)
        self.support = support
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "num_objects=" + str(self.num_objects)
        s += ", omega=" + str(self.omega)
        s += ", cdf=" + str(self.cdf)
        s += ", support=" + str(self.support)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s
    
    def sample(self, n=1):
        u = np.random.rand(n)
        rez = np.zeros([n])
        for ii in range(0, n):
            jj = 0
            while u[ii] > self.cdf[jj]:
                jj += 1
            rez[ii] = self.support[jj]
        return Sample(rez.astype(np.int32))
    
    def log_pdf(self, X):
        return None
