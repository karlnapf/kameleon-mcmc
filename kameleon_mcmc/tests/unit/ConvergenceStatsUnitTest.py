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

from numpy import sum
from numpy.random import randn
import unittest

from kameleon_mcmc.tools.ConvergenceStats import ConvergenceStats


class ConvergenceStatsUnitTest(unittest.TestCase):
    def test_wrong_input_type_x(self):
        self.assertRaises(TypeError, ConvergenceStats.autocorr, None)
        
    def test_wrong_input_type_normalise(self):
        x = randn(100)
        self.assertRaises(TypeError, ConvergenceStats.autocorr, x, None)
        
    def test_wrong_array_shape_x(self):
        x = randn(100, 1)
        self.assertRaises(ValueError, ConvergenceStats.autocorr, x)
        
    def test_normaliser(self):
        x = randn(100)
        _, z = ConvergenceStats.autocorr(x)
        self.assertEqual(z, sum(x ** 2))
        
    def test_normalise_param_true(self):
        x = randn(100)
        c, _ = ConvergenceStats.autocorr(x, True)
        self.assertEqual(c[0], 1.)
        
    def test_normalise_param_false(self):
        x = randn(100)
        c, z = ConvergenceStats.autocorr(x, False)
        self.assertEqual((c / z)[0], 1.)
        
    def test_normalise_param_default_is_true(self):
        x = randn(100)
        c, _ = ConvergenceStats.autocorr(x)
        self.assertEqual(c[0], 1.)
        
if __name__ == "__main__":
    unittest.main()
