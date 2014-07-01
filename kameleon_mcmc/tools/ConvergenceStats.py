"""
Copyright (c) 2013-2014 Heiko Strathmann, Dino Sejdinovic
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
from kameleon_mcmc.tools.GenericTests import GenericTests
import numpy as np


class ConvergenceStats():
    '''
    Class that implements various convergence statistics for Markov chains
    '''

    @staticmethod
    def autocorr(x):
        """
        Computes the ( normalised) auto-correlation function of a
        one dimensional sequence of numbers.
        
        Utilises the numpy correlate function that is based on an efficient
        convolution implementation.
        
        Inputs:
        x - one dimensional numpy array
        
        Outputs:
        Vector of autocorrelation values for a lag from zero to max possible
        """
        
        GenericTests.check_type(x, "x", np.ndarray, 1)
        
        # normalise, compute norm
        xunbiased = x - np.mean(x)
        xnorm = np.sum(xunbiased ** 2)
        
        # convolve with itself
        acor = np.correlate(xunbiased, xunbiased, mode='same')
        
        # use only second half, normalise
        acor = acor[len(acor) / 2:] / xnorm
        
        return acor
    
    @staticmethod
    def gelman_rubin(x):
        """ Returns estimate of R for a set of traces.
    
        The Gelman-Rubin diagnostic tests for lack of convergence by comparing
        the variance between multiple chains to the variance within each chain.
        If convergence has been achieved, the between-chain and within-chain
        variances should be identical. To be most effective in detecting evidence
        for nonconvergence, each chain should have been initialized to starting
        values that are dispersed relative to the target distribution.
    
        Parameters
        ----------
        x : array-like
          A two-dimensional array containing the parallel traces (minimum 2)
          of some stochastic parameter.
    
        Returns
        -------
        Rhat : float
          Return the potential scale reduction factor, :math:`\hat{R}`
    
        Notes
        -----
    
        The diagnostic is computed by:
    
          .. math:: \hat{R} = \frac{\hat{V}}{W}
    
        where :math:`W` is the within-chain variance and :math:`\hat{V}` is
        the posterior variance estimate for the pooled traces. This is the
        potential scale reduction factor, which converges to unity when each
        of the traces is a sample from the target posterior. Values greater
        than one indicate that one or more chains have not yet converged.
    
        References
        ----------
        Brooks and Gelman (1998)
        Gelman and Rubin (1992)
        
        Copyright
        ---------
        Taken from the pymc package 2.3
        """
    
        if np.shape(x) < (2,):
            raise ValueError(
                'Gelman-Rubin diagnostic requires multiple chains of the same length.')
    
        m, n = np.shape(x)
    
        # Calculate between-chain variance
        B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)
    
        # Calculate within-chain variances
        W = np.sum(
            [(x[i] - xbar) ** 2 for i,
             xbar in enumerate(np.mean(x,
                                       1))]) / (m * (n - 1))
    
        # (over) estimate of variance
        s2 = W * (n - 1) / n + B_over_n
    
        # Pooled posterior variance estimate
        V = s2 + B_over_n / m
    
        # Calculate PSRF
        R = V / W
    
        return R
