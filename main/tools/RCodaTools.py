"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from rpy2 import robjects

class RCodaTools(object):
    
    @staticmethod
    def ess_coda(data):
        """
        Computes the effective samples size of a 1d-array using R-coda via
        an external R call. The python package rpy2 and the R-library
        "library(coda)" have to be installed. Inspired by Charles Blundell's
        neat little python script :)
        """
        robjects.r('library(coda)')
        r_ess = robjects.r['effectiveSize']
        data = robjects.r.matrix(robjects.FloatVector(data), nrow=len(data))
        return r_ess(data)[0]