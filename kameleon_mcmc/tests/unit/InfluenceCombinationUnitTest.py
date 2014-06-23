"""
Copyright (c) 2013-2014 Heiko Strathmann
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 *
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 *
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the author.
"""

from numpy import zeros, ones, asarray, array
import numpy
from numpy.random import rand, randn, randint
import unittest

from kameleon_mcmc.distribution.InfluenceCombination import InfluenceCombination
from kameleon_mcmc.distribution.Distribution import Sample


class InfluenceCombinationUnitTest(unittest.TestCase):
    def test_contructor_wrong_W_type_float(self):
        W = 0
        biasx = array([0, 0])
        biash = array([0, 0])
        self.assertRaises(TypeError, InfluenceCombination, W, biasx, biash)
        
    def test_contructor_wrong_W_type_none(self):
        W = None
        biasx = array([0, 0])
        biash = array([0, 0])
        self.assertRaises(TypeError, InfluenceCombination, W, biasx, biash)
        
    def test_contructor_wrong_biasx_type_float(self):
        d = 10 #number of visible units
        h = 3 #number of hidden units
        W = randn(h,d)
        biasx = 0
        biash = randn(h)
        self.assertRaises(TypeError, InfluenceCombination, W, biasx, biash)
        
    def test_contructor_wrong_biasx_type_none(self):
        d = 10 #number of visible units
        h = 3 #number of hidden units
        W = randn(h,d)
        biasx = None
        biash = randn(h)
        self.assertRaises(TypeError, InfluenceCombination, W, biasx, biash)
        
    def test_contructor_wrong_biasx_dim(self):
        d = 10 #number of visible units
        h = 3 #number of hidden units
        W = randn(h,d)
        biasx = randn(d,1)
        biash = randn(h)
        self.assertRaises(ValueError, InfluenceCombination, W, biasx, biash)
        
    def test_contructor_W_and_biasx_dim_not_matching(self):
        d = 10 #number of visible units
        h = 3 #number of hidden units
        W = randn(h,d)
        biasx = randn(d+1)
        biash = randn(h)
        self.assertRaises(ValueError, InfluenceCombination, W, biasx, biash)
        
    def test_log_pdf_many_no_checks(self):
        d = 10 #number of visible units
        h = 3 #number of hidden units
        W = randn(h,d)
        biasx = randn(d)
        biash = randn(h)
        n = 20  #number of samples
        X = rand(n,d)<0.5
        dist = InfluenceCombination(W,biasx,biash)
        log_probs = dist.log_pdf(X)
        #print log_probs
        
if __name__ == "__main__":
    unittest.main()
