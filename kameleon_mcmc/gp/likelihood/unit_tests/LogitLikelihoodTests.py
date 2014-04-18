"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from kameleon_mcmc.gp.LogitLikelihood import LogitLikelihood
from numpy.linalg.linalg import norm
from numpy.ma.core import asarray
from numpy.random import randn, randint
import unittest

class Test(unittest.TestCase):

    def test_log_lik_vector_multiple1(self):
        n=100
        y=randint(0,2,n)*2-1
        f=randn(n)
        
        lik=LogitLikelihood()
        multiple=lik.log_lik_vector_multiple(y, f.reshape(1,n))
        single=lik.log_lik_vector(y, f)
        
        self.assertLessEqual(norm(single-multiple), 1e-10)
        
    def test_log_lik_vector_multiple2(self):
        n=100
        y=randint(0,2,n)*2-1
        F=randn(10,n)
        
        lik=LogitLikelihood()
        multiples=lik.log_lik_vector_multiple(y, F)
        singles=asarray([lik.log_lik_vector(y, f) for f in F])
        
        self.assertLessEqual(norm(singles-multiples), 1e-10)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()