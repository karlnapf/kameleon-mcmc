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

from numpy import zeros, asarray, tanh
import numpy
from numpy.linalg.linalg import norm
from numpy.random import randint
import unittest

from kameleon_mcmc.kernel.HypercubeKernel import HypercubeKernel


class HypercubeKernelUnitTest(unittest.TestCase):
    def test_contructor_wrong_gamma_type(self):
        self.assertRaises(TypeError, HypercubeKernel, gamma=None)
        
    def test_contructor_gamma(self):
        gamma = 1.2
        k = HypercubeKernel(gamma=gamma)
        self.assertEqual(gamma, k.gamma)
        
    def test_kernel_wrong_X_type(self):
        k = HypercubeKernel(1.)
        X = None
        Y = zeros((2, 2))
        self.assertRaises(TypeError, k.kernel, X, Y)
        
    def test_kernel_wrong_X_array_dtype1(self):
        k = HypercubeKernel(1.)
        X = zeros(2, dtype=numpy.float64)
        Y = zeros((2, 2), dtype=numpy.bool8)
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_wrong_X_array_dtype2(self):
        k = HypercubeKernel(1.)
        X = zeros(2, dtype=numpy.int64)
        Y = zeros((2, 2), dtype=numpy.bool8)
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_wrong_X_array_dimension1(self):
        k = HypercubeKernel(1.)
        X = zeros(2)
        Y = zeros((2, 2))
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_wrong_X_array_dimension2(self):
        k = HypercubeKernel(1.)
        X = zeros((2, 3, 2))
        Y = zeros((2, 2))
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_wrong_Y_type(self):
        k = HypercubeKernel(1.)
        X = zeros((2, 2))
        Y = 3
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_wrong_Y_array_dimension1(self):
        k = HypercubeKernel(1.)
        X = zeros((2, 3))
        Y = zeros(2)
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_wrong_Y_array_dimension2(self):
        k = HypercubeKernel(1.)
        X = zeros((2, 3))
        Y = zeros((2, 3, 4))
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_wrong_Y_array_dtype1(self):
        k = HypercubeKernel(1.)
        X = zeros(2, dtype=numpy.bool8)
        Y = zeros((2, 2), dtype=numpy.float64)
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_wrong_Y_array_dtype2(self):
        k = HypercubeKernel(1.)
        X = zeros(2, dtype=numpy.bool8)
        Y = zeros((2, 2), dtype=numpy.int64)
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_wrong_X_Y_same_dimension1(self):
        k = HypercubeKernel(1.)
        X = zeros((2, 3))
        Y = zeros((2, 4))
        self.assertRaises(ValueError, k.kernel, X, Y)
        
    def test_kernel_X_alone_type(self):
        k = HypercubeKernel(1.)
        n = 3
        d = 2
        X = zeros((n, d), dtype=numpy.bool8)
        K = k.kernel(X)
        self.assertEqual(type(K), numpy.ndarray)
    
    def test_kernel_X_alone_dimension(self):
        k = HypercubeKernel(1.)
        n = 3
        d = 2
        X = zeros((n, d), dtype=numpy.bool8)
        K = k.kernel(X)
        self.assertEqual(K.shape, (n, n))
        
    def test_kernel_X_alone_dtype(self):
        gamma = .2
        k = HypercubeKernel(gamma)
        X = asarray([[0]], dtype=numpy.bool8)
        K = k.kernel(X)
        self.assertEqual(K.dtype, numpy.float)
        
    def test_kernel_X_Y_dimension(self):
        k = HypercubeKernel(1.)
        n_X = 3
        n_Y = 4
        d = 2
        X = zeros((n_X, d), dtype=numpy.bool8)
        Y = zeros((n_Y, d), dtype=numpy.bool8)
        K = k.kernel(X, Y)
        self.assertEqual(K.shape, (n_X, n_Y))
    
    def test_kernel_X_one_point_one(self):
        gamma = .2
        k = HypercubeKernel(gamma)
        X = asarray([[1]], dtype=numpy.bool8)
        K = k.kernel(X)
        self.assertEqual(K[0, 0], 1.)
        
    def test_kernel_X_one_point_zero(self):
        gamma = .2
        k = HypercubeKernel(gamma)
        X = asarray([[0]], dtype=numpy.bool8)
        K = k.kernel(X)
        self.assertEqual(K[0, 0], 1.)
        
    def test_kernel_X_two_points_fixed(self):
        gamma = .2
        k = HypercubeKernel(gamma)
        X = asarray([[1, 0], [1, 1]], dtype=numpy.bool8)
        K = zeros((2, 2))
        for i in range(2):
            for j in range(2):
                dist = sum(X[i] != X[j])
                K[i, j] = tanh(gamma) ** dist
        self.assertAlmostEqual(norm(K - k.kernel(X)), 0)
        
    def test_kernel_X_many_points_random(self):
        gamma = .2
        n_X = 4
        d = 5
        num_runs = 100
        k = HypercubeKernel(gamma)
        
        for _ in range(num_runs):
            X = randint(0, 2, (n_X, d)).astype(numpy.bool8)
            K = zeros((n_X, n_X))
            for i in range(n_X):
                for j in range(n_X):
                    dist = sum(X[i] != X[j])
                    K[i, j] = tanh(gamma) ** dist
            self.assertAlmostEqual(norm(K - k.kernel(X)), 0)
            
    def test_kernel_X_Y_one_point_same(self):
        gamma = .2
        k = HypercubeKernel(gamma)
        X = asarray([[1]], dtype=numpy.bool8)
        Y = asarray([[1]], dtype=numpy.bool8)
        K = k.kernel(X, Y)
        self.assertEqual(K[0, 0], 1.)
        
    def test_kernel_X_Y_one_point_different(self):
        gamma = .2
        k = HypercubeKernel(gamma)
        X = asarray([[1]], dtype=numpy.bool8)
        Y = asarray([[0]], dtype=numpy.bool8)
        K = k.kernel(X, Y)
        self.assertEqual(K[0, 0], tanh(gamma))
        
    def test_kernel_X_Y_many_points_random(self):
        gamma = .2
        n_X = 4
        n_Y = 3
        d = 5
        num_runs = 100
        k = HypercubeKernel(gamma)
        
        for _ in range(num_runs):
            X = randint(0, 2, (n_X, d)).astype(numpy.bool8)
            Y = randint(0, 2, (n_Y, d)).astype(numpy.bool8)
            K = zeros((n_X, n_Y))
            for i in range(n_X):
                for j in range(n_Y):
                    dist = sum(X[i] != Y[j])
                    K[i, j] = tanh(gamma) ** dist
            self.assertAlmostEqual(norm(K - k.kernel(X,Y)), 0)

        
if __name__ == "__main__":
    unittest.main()
