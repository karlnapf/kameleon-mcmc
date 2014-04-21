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

from numpy import zeros
from numpy.random import rand
import unittest

from kameleon_mcmc.distribution.Bernoulli import Bernoulli
from kameleon_mcmc.kernel.HypercubeKernel import HypercubeKernel
from kameleon_mcmc.mcmc.samplers.DiscreteKameleon import DiscreteKameleon


class DiscreteKameleonUnitTest(unittest.TestCase):
    def test_contructor_wrong_distribution_type(self):
        dimension = 3
        ps = rand(dimension)
        distribution = Bernoulli(ps)
        kernel = HypercubeKernel(1.)
        Z = zeros((2, distribution.dimension))
        self.assertRaises(TypeError, DiscreteKameleon, None, kernel, Z)
        
    def test_contructor_wrong_kernel_type(self):
        dimension = 3
        ps = rand(dimension)
        distribution = Bernoulli(ps)
        Z = zeros((2, distribution.dimension))
        self.assertRaises(TypeError, DiscreteKameleon, distribution, None, Z)
        
    def test_contructor_wrong_Z_type(self):
        dimension = 3
        ps = rand(dimension)
        distribution = Bernoulli(ps)
        kernel = HypercubeKernel(1.)
        self.assertRaises(TypeError, DiscreteKameleon, distribution, kernel, None)
        
    def test_contructor_wrong_Z_array_dimension_too_small(self):
        dimension = 3
        ps = rand(dimension)
        distribution = Bernoulli(ps)
        kernel = HypercubeKernel(1.)
        Z = zeros(distribution.dimension)
        self.assertRaises(ValueError, DiscreteKameleon, distribution, kernel, Z)
        
    def test_contructor_wrong_Z_array_dimension_too_large(self):
        dimension = 3
        ps = rand(dimension)
        distribution = Bernoulli(ps)
        kernel = HypercubeKernel(1.)
        Z = zeros((2, distribution.dimension, 3))
        self.assertRaises(ValueError, DiscreteKameleon, distribution, kernel, Z)
        
    def test_contructor_wrong_Z_dimension_too_small(self):
        dimension = 3
        ps = rand(dimension)
        distribution = Bernoulli(ps)
        kernel = HypercubeKernel(1.)
        Z = zeros((2, distribution.dimension - 1))
        self.assertRaises(ValueError, DiscreteKameleon, distribution, kernel, Z)
        
    def test_contructor_wrong_Z_dimension_too_big(self):
        dimension = 3
        ps = rand(dimension)
        distribution = Bernoulli(ps)
        kernel = HypercubeKernel(1.)
        Z = zeros((2, distribution.dimension + 1))
        self.assertRaises(ValueError, DiscreteKameleon, distribution, kernel, Z)
        
    def test_contructor(self):
        dimension = 3
        ps = rand(dimension)
        distribution = Bernoulli(ps)
        kernel = HypercubeKernel(1.)
        Z = zeros((2, distribution.dimension))
        sampler = DiscreteKameleon(distribution, kernel, Z)
        self.assertEqual(sampler.distribution, distribution)
        self.assertEqual(sampler.kernel, kernel)
        self.assertTrue(sampler.Z is Z)
        
if __name__ == "__main__":
    unittest.main()
