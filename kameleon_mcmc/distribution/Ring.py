"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from kameleon_mcmc.distribution.Distribution import Distribution
from kameleon_mcmc.distribution.Flower import Flower
from numpy import sqrt

class Ring(Flower):
    def __init__(self, variance=0.05, radius=3.5, dimension=2):
        Flower.__init__(self, 0, 1, variance, radius, dimension)

    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "variance=" + str(self.variance)
        s += ", radius=" + str(self.radius)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s

    def get_plotting_bounds(self):
        if self.dimension == 2:
            value = self.radius + self.amplitude + 3 * sqrt(self.variance)
            return [(-value, value) for _ in range(2)]
        else:
            return Flower.get_plotting_bounds(self)