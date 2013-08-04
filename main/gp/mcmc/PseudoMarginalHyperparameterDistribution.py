"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from main.distribution.Distribution import Distribution
from matplotlib.pyplot import subplot, imshow, figure, plot, contour, pcolor, \
    show, hold
from numpy.core.function_base import linspace
from numpy.ma.core import shape, zeros, exp, asarray, array, reshape
from shogun.Features import BinaryLabels, RealFeatures
from shogun.GaussianProcess import LaplacianInferenceMethod, LogitLikelihood, \
    ZeroMean
from shogun.Kernel import GaussianKernel, GaussianARDKernel
from modshogun import RealFeatures, BinaryLabels, GaussianKernel, GaussianARDKernel, Math
from modshogun import ProbitLikelihood, ZeroMean, LaplacianInferenceMethod, GaussianProcessBinaryClassification
import itertools


class PseudoMarginalHyperparameterDistribution(Distribution):
    """
    Class to represent a GP's marginal posterior distribution of hyperparameters
    
    p(theta|y) \propto p(y|theta) p(theta)
    
    as an MCMC target. The p(y|theta) function is an unbiased estimate
    """
    def __init__(self, X, y, n_importance, prior, ridge):
        Distribution.__init__(self, dimension=shape(X)[1])
        
        self.n_importance=n_importance
        self.prior=prior
        self.ridge=ridge
        self.X=X
        self.y=y
    
    def log_pdf(self, thetas):
        assert(len(shape(thetas)) == 2)
        assert(shape(thetas)[1] == self.dimension)
        
        result=zeros(len(thetas))
        for i in range(len(thetas)):
            # perform inference for now
            data=self.X
            lab=self.y
#            ns_test=[100,40]
#            P=linspace(data[:,0].min(), data[:,0].max(), ns_test[0])
#            Q=linspace(data[:,1].min(), data[:,1].max(), ns_test[1])
#            data_test=asarray(list(itertools.product(P, Q)))
            
            labels=BinaryLabels(lab)
            feats_train=RealFeatures(data.T)
#            feats_test=RealFeatures(data_test.T)
            
            kernel=GaussianARDKernel(10,1)
            kernel.set_weights(exp(thetas[i]))
            mean=ZeroMean()
            likelihood=ProbitLikelihood()
            inference=LaplacianInferenceMethod(kernel, feats_train, mean, labels, likelihood)
            inference.set_scale(exp(0))
            
            log_ml_la=-inference.get_negative_marginal_likelihood()
            log_ml_estimate=inference.get_log_ml_estimate(10000, 1e-5)
            print "theta:", kernel.get_weights()
            print "LA ml:", log_ml_la
            print "estimated ml:", log_ml_estimate
#            
#            gp = GaussianProcessBinaryClassification(inference)
#            gp.train()
#            predictions=gp.apply_binary(feats_test)
#            Y_test=predictions.get_values()
#            Y_test=reshape(Y_test, (ns_test[0],ns_test[1]))
#            
#            subplot(121)
#            kernel.init(feats_train, feats_train)
#            imshow(kernel.get_kernel_matrix())
#            
#            subplot(122)
#            imshow(inference.get_posterior_approximation_covariance())
#            
#            figure()
#
#            contour(P,Q,Y_test.T, levels=[0])
#            pcolor(P,Q,Y_test.T)
#            show()
            
            # prior is also in log-domain, so no exp call
            log_prior=self.prior.log_pdf(thetas[i].reshape(1,len(thetas[i])))
            result[i]=log_ml_estimate+log_prior
            
            print "log marginal likelihood for (log-domain) theta=", thetas[i], "is", log_ml_estimate, "under LA:", log_ml_la
#            print "log marginal likelihood for theta=", theta, "is under LA:", -self.inf.get_negative_marginal_likelihood()
            
        return result
        