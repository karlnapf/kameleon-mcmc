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
from numpy import exp
from numpy.random import randint
from scipy.special import binom
import unittest

from kameleon_mcmc.tools.HelperFunctions import HelperFunctions


class BernoulliUnitTest(unittest.TestCase):
    def test_log_binom_coeff_1(self):
        n = 2
        k = 1
        self.assertEqual(round(exp(HelperFunctions.log_bin_coeff(n, k))), binom(n, k))
        
    def test_log_binom_coeff_2(self):
        n = 0
        k = 1
        self.assertEqual(round(exp(HelperFunctions.log_bin_coeff(n, k))), binom(n, k))
        
    def test_log_binom_coeff_3(self):
        n = 0
        k = 0
        self.assertEqual(round(exp(HelperFunctions.log_bin_coeff(n, k))), binom(n, k))
        
    def test_log_binom_coeff_4(self):
        n = 1
        k = 0
        self.assertEqual(round(exp(HelperFunctions.log_bin_coeff(n, k))), binom(n, k))
        
    def test_log_binom_coeff_5(self):
        n = 2
        k = 3
        self.assertEqual(round(exp(HelperFunctions.log_bin_coeff(n, k))), binom(n, k))
        
    def test_log_binom_coeff_many(self):
        for _ in range(100):
            n = randint(1, 10)
            k = randint(0, n)
            self.assertEqual(round(exp(HelperFunctions.log_bin_coeff(n, k))), round(binom(n, k)))
    
if __name__ == "__main__":
    unittest.main()
