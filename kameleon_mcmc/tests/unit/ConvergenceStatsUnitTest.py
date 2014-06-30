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
import unittest

from kameleon_mcmc.tools.ConvergenceStats import ConvergenceStats
import numpy as np


class ConvergenceStatsUnitTest(unittest.TestCase):
    def test_wrong_input_type_x(self):
        self.assertRaises(TypeError, ConvergenceStats.autocorr, None)
        
    def test_wrong_input_type_normalise(self):
        x = np.random.randn(100)
        self.assertRaises(TypeError, ConvergenceStats.autocorr, x, None)
        
    def test_wrong_array_shape_x(self):
        x = np.random.randn(100, 1)
        self.assertRaises(ValueError, ConvergenceStats.autocorr, x)
        
    def test_normalisation(self):
        x = np.random.randn(100)
        acorr = ConvergenceStats.autocorr(x)
        self.assertEqual(acorr[0], 1.)
        
    def test_result(self):
        x = np.random.randn(100)
        y = x - np.mean(x)
        ynorm = np.sum(y ** 2)
        acor = np.correlate(y, y, mode='same')[50:] / ynorm
        self.assertTrue(all(acor == ConvergenceStats.autocorr(x)))
        
if __name__ == "__main__":
    unittest.main()
