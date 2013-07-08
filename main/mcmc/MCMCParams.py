"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

class MCMCParams(object):
    def __init__(self, start, num_iterations=80000, burnin=60000):
        self.num_iterations = num_iterations
        self.burnin = burnin
        self.start = start
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "num_iteratons="+ str(self.num_iterations)
        s += ", burnin="+ str(self.burnin)
        s += ", start="+ str(self.start)
        s += "]"
        return s