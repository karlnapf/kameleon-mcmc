"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy import int32, array
from numpy.ma.core import shape, zeros, exp
from shogun.Classifier import LaplacianInferenceMethod, LogitLikelihood, \
    ZeroMean
from shogun.Features import BinaryLabels, RealFeatures
from shogun.Kernel import DiffusionKernel

from main.distribution.Distribution import Distribution


# tell shogun to use 1 thread only (extra to python)
num_threads=1
print "Using Shogun with %d threads" % num_threads
ZeroMean().parallel.set_num_threads(num_threads)

class PseudoMarginalHyperparameterDistributionDiffusion(Distribution):
    """
    Class to represent a GP's marginal posterior distribution of hyperparameters
    
    p(theta|y) \propto p(y|theta) p(theta)
    
    as an MCMC target. The p(y|theta) function is an unbiased estimate.
    Hyperparameters are the length scales of a Gaussian ARD kernel.
    
    Uses the Shogun machine learning toolbox for GP inference.
    """
    def __init__(self, X, y, n_importance, prior, ridge=None):
        Distribution.__init__(self, dimension=shape(X)[1])
        
        self.n_importance=n_importance
        self.prior=prior
        self.ridge=ridge
        self.X=X
        self.y=y
        
        # compute alphabet sizes based on number of unique elements for each
        # covariate in the cateogrical input data represented as reals
        self.alphabet_sizes=array([len(set(X[:,i])) for i in range(X.shape[1])],
                                  dtype=int32)
    
    def log_pdf(self, thetas):
        assert(len(shape(thetas)) == 2)
        assert(shape(thetas)[1] == self.dimension)
        
        result=zeros(len(thetas))
        for i in range(len(thetas)):
            labels=BinaryLabels(self.y)
            feats_train=RealFeatures(self.X.T)

            # ARD: set set theta, which is in log-scale, as kernel weights
            betas=exp(thetas[i])
            kernel=DiffusionKernel(betas, self.alphabet_sizes)
            
            mean=ZeroMean()
            likelihood=LogitLikelihood()
            inference=LaplacianInferenceMethod(kernel, feats_train, mean, labels, likelihood)
            
            # fix kernel scaling for now
            inference.set_scale(exp(0))
            
            if self.ridge is not None:
                log_ml_estimate=inference.get_marginal_likelihood_estimate(self.n_importance, self.ridge)
            else:
                log_ml_estimate=inference.get_marginal_likelihood_estimate(self.n_importance)
            
            # prior is also in log-domain, so no exp of theta
            log_prior=self.prior.log_pdf(thetas[i].reshape(1,len(thetas[i])))
            result[i]=log_ml_estimate+log_prior
            
        return result
        
