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

from numpy import zeros, ones, asarray, mean, log, sum
import numpy
from numpy.random import rand, randint
import unittest

from kameleon_mcmc.distribution.Bernoulli import Bernoulli
from kameleon_mcmc.distribution.Distribution import Sample


class BernoulliUnitTest(unittest.TestCase):
    def test_contructor_wrong_ps_type1(self):
        ps = 0
        self.assertRaises(TypeError, Bernoulli, ps)
        
    def test_contructor_wrong_ps_type2(self):
        ps = None
        self.assertRaises(TypeError, Bernoulli, ps)

    def test_contructor_wrong_ps_dim(self):
        ps = zeros(0)
        self.assertRaises(ValueError, Bernoulli, ps)
        
    def test_contructor_wrong_ps_range_0(self):
        ps = zeros(0)
        self.assertRaises(ValueError, Bernoulli, ps)
        
    def test_contructor_wrong_ps_range_1(self):
        ps = ones(0)
        self.assertRaises(ValueError, Bernoulli, ps)
        
    def test_contructor_correct_dim(self):
        ps = rand(4)
        b = Bernoulli(ps)
        self.assertEqual(ps.shape, b.ps.shape)
        
    def test_contructor_correct_values(self):
        ps = rand(4)
        b = Bernoulli(ps)
        self.assertTrue(all(ps == b.ps))
    
    def test_sample_wrong_n_zero(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertRaises(ValueError, b.sample, 0)
    
    def test_sample_wrong_n_sameller_zero(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertRaises(ValueError, b.sample, -1)
    
    def test_sample_wrong_n_type_none(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertRaises(TypeError, b.sample, None)
        
    def test_sample_wrong_n_type_float(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertRaises(TypeError, b.sample, float(1.))
        
    def test_sample_type(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertTrue(isinstance(b.sample(1), Sample))
        
    def test_sample_samples_dtype(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertEqual(b.sample(1).samples.dtype, numpy.bool8)
        
    def test_sample_dim(self):
        n = 3
        d = 2
        p = asarray(rand(d))
        b = Bernoulli(p)
        s = b.sample(n)
        self.assertEqual(s.samples.shape, (n, d))
        
    def test_sample_mean_values(self):
        n = 10000
        d = 3
        runs = 100
        
        for _ in range(runs):
            ps = asarray(rand(d))
            b = Bernoulli(ps)
            s = b.sample(n)
            for i in range(d):
                self.assertAlmostEqual(mean(s.samples[:, i]), ps[i], delta=0.05)
                
    def test_log_pdf_wrong_type_none(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertRaises(TypeError, b.log_pdf, None)
        
    def test_log_pdf_wrong_type_float(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertRaises(TypeError, b.log_pdf, float(1.))
        
    def test_log_pdf_wrong_array_dimension_1(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertRaises(TypeError, b.log_pdf, zeros(1))
        
    def test_log_pdf_wrong_array_dimension_3(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertRaises(TypeError, b.log_pdf, zeros(3))
        
    def test_log_pdf_wrong_dimension(self):
        p = asarray([0.5])
        b = Bernoulli(p)
        self.assertRaises(ValueError, b.log_pdf, zeros((1, 2)))
        
    def test_log_pdf_type(self):
        p = asarray([0.3])
        b = Bernoulli(p)
        self.assertEqual(type(b.log_pdf(asarray([[1]], dtype=numpy.bool8))), numpy.ndarray)
    
    def test_log_pdf_array_dimension(self):
        d = 3
        n = 2
        ps = asarray(rand(d))
        b = Bernoulli(ps)
        X = randint(0, 2, (n, d)).astype(numpy.bool8)
        self.assertEqual(b.log_pdf(X).shape, (n,))
        
    def test_log_pdf_success_single(self):
        d = 1
        ps = asarray(rand(d))
        b = Bernoulli(ps)
        X = asarray([[1]], dtype=numpy.bool8)
        self.assertEqual(b.log_pdf(X), log(ps[0]))
        
    def test_log_pdf_failure_single(self):
        d = 1
        ps = asarray(rand(d))
        b = Bernoulli(ps)
        X = asarray([[0]], dtype=numpy.bool8)
        self.assertEqual(b.log_pdf(X), log(1 - ps[0]))
        
    def test_log_pdf_success_multiple(self):
        d = 4
        n = 3
        num_runs = 100
        
        for _ in range(num_runs):
            ps = rand(d)
            b = Bernoulli(ps)
            X = randint(0, 2, (n, d)).astype(numpy.bool8)
            
            # naive computation of log pdf
            expected = zeros((n, d))
            for i in range(n):
                for j in range(d):
                    expected[i, j] = log(ps[j]) if X[i, j] == 1 else log(1 - ps[j])
                    
            self.assertTrue(all(sum(expected, 1) == b.log_pdf(X)))
        
if __name__ == "__main__":
    unittest.main()
