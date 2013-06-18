from main.distribution.Gaussian import Gaussian
from numpy.linalg.linalg import cholesky
from numpy.ma.core import zeros, asarray

class GaussianProcess(object):
    def __init__(self, y, X, covariance, likelihood, hyper_prior, theta=None):
        """
        y - data (labels)
        X - covariates
        """
        self.y = y
        self.X = X
        self.covariance = covariance
        self.likelihood = likelihood
        self.hyper_prior=hyper_prior
        self.theta=None
        if theta is not None:
            self.update_gp_prior(theta)

    def update_gp_prior(self, theta):
        self.covariance.set_theta(theta)
        L=cholesky(self.covariance.compute(self.X))
        self.gp_prior=Gaussian(mu=zeros(len(self.y)), Sigma=L, is_cholesky=True)
        
    def log_lik_f_given_theta(self, f, theta=None):
        """
        Computes log(p(f|theta))
        
        f can be 1d or 2d array
        """
        assert(len(f)==len(self.y))
        if theta is not None:
            self.update_gp_prior(theta)
        
        return self.gp_prior.log_pdf(f)
        
    def log_lik_theta(self, theta=None):
        """
        Computes log(p(theta))
        
        theta can be 1d or 2d array or None (then last one is used)
        """
        if theta is None:
            theta=asarray([self.covariance.get_theta(), self.likelihood.get_theta()])
        
        return self.hyper_prior.log_lik(theta)
          
    def log_lik_y_given_f(self, f):
        """
        Computes log(p(y|f))
        """
        return self.likelihood.log_lik(self.y, f)
    
    def log_lik(self, f, theta=None):
        """
        Computes log(p(y,f,theta))
        """
        prior=self.log_lik_theta(theta)
        latent=self.log_lik_f_given_theta(f, theta)
        lik=self.log_lik(f, theta)
        
        return prior+latent+lik
    
