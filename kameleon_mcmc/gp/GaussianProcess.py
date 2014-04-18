"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from abc import abstractmethod
from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.gp.GPTools import GPTools
from kameleon_mcmc.gp.covariance.SquaredExponentialCovariance import \
    SquaredExponentialCovariance
from kameleon_mcmc.gp.inference.LaplaceApproximation import LaplaceApproximation
from kameleon_mcmc.gp.likelihood.LogitLikelihood import LogitLikelihood
from numpy.lib.function_base import delete
from numpy.linalg.linalg import solve
from numpy.ma.core import zeros, reshape, asarray, log, exp, shape, array, mean
from numpy.ma.extras import hstack, cov

class GaussianProcess(object):
    def __init__(self, y, X, covariance, likelihood):
        """
        y - data (labels)
        X - covariates
        """
        self.y = y
        self.X = X
        self.covariance = covariance
        self.likelihood = likelihood
        self.K = self.covariance.compute(self.X)

    def get_gp_prior(self):
        """
        Returns GP prior N(0,K), only possible do if K is psd
        """
        return Gaussian(zeros(len(self.K)), self.K, is_cholesky=False)
        
    def log_prior(self, f):
        """
        Computes log(p(f)), only possible do if K is psd
        
        f - 1d vector
        """
        assert(len(f) == len(self.y))
        f_2d = reshape(f, (1, len(f)))
        return self.get_gp_prior().log_pdf(f_2d)
    
    def log_prior_grad_vector(self, f):
        return -solve(self.K, f)
    
    def log_likelihood_grad_vector(self, f):
        return self.likelihood.log_lik_grad_vector(self.y, f)
        
    def log_likelihood(self, f):
        """
        Computes log(p(y|f))
        """
        return sum(self.likelihood.log_lik_vector(self.y, f))
    
    def log_likelihood_multiple(self, F):
        all=self.likelihood.log_lik_vector_multiple(self.y, F)
        return asarray([sum(a) for a in all])
    
    def log_posterior_unnormalised(self, f):
        """
        Computes log(p(y,f))=log(p(y|f)+log(p(f)), only possible do if K is psd
        """
        lik = self.log_likelihood(f)
        prior = self.log_prior(f)
        
        return prior + lik
    
    def log_posterior_grad_vector(self, f):
        lik = self.log_likelihood_grad_vector(f)
        prior = self.log_prior_grad_vector(f)
        
        return prior + lik

    def log_ml_estimate(self, proposal, n=1):
        """
        Computes an estimate for the marginal likelihood p(y) using importance
        sampling the provided proposal distribution
        """
        
        prior = self.get_gp_prior()
        
        # sample from proposal
        samples = proposal.sample(n).samples
        
        # compute log likelihoods of samples
        log_likelihood=self.log_likelihood_multiple(samples)
        log_prior = prior.log_pdf(samples)
        log_proposal = proposal.log_pdf(samples)
        
        # compute estimate of marginal likelihood, log sum exp trick
        X=log_likelihood+log_prior-log_proposal
        
        return GPTools.log_mean_exp(X)
    
    def gen_num_hyperparameters(self):
        return self.covariance.get_num_hyperparameters() + \
            self.likelihood.get_num_hyperparameters()
    
    def get_hyperparameters(self):
        theta=self.covariance.get_hyperparameters()
        theta=hstack((theta, self.likelihood.get_hyperparameters()))

        return theta
    
    def set_hyperparameters(self, theta):
        num_cov=self.covariance.gen_num_hyperparameters()
        self.covariance.set_hyperparameters(theta[:num_cov])
        self.likelihood.set_hyperparameters(theta[num_cov:])

