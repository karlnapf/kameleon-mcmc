"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Parts translated from Matlab GPML toolbox of the "Gaussian Processes in Machine
Learning" book by Rasmussen and Williams
"""
from kameleon_mcmc.distribution.Gaussian import Gaussian
from numpy.core.numeric import inf
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import cholesky
from numpy.ma.core import zeros, sqrt, shape, asarray
from scipy import integrate
from scipy.linalg.basic import solve_triangular
from scipy.stats import norm

class LaplaceApproximation(object):
    def __init__(self, gp, newton_step=1.0, newton_epsilon=1e-5, \
                 newton_max_iterations=20, newton_start=None):
        """
        gp - underlying Gaussian process
        newton_step - starting step size, if the objective function is not
                      increased after a step, the step is discarded and step size
                      is halfed
        newton_epsilon - epsilon to terminate optimisation
        newton_max_iterations - maximum number of steps
        newton_start - optional starting point, useful if mode has to be found multiple
                       times for slightly varying data
        """
        dim = len(gp.K)
        
        assert(newton_step > 0)
        assert(newton_epsilon > 0)
        assert(newton_max_iterations > 0)
        
        if newton_start is not None:
            assert(len(shape(newton_start)) == 1)
            assert(len(newton_start) == dim)
        
        self.gp = gp
        self.newton_step = newton_step
        self.newton_epsilon = newton_epsilon
        self.newton_max_iterations = newton_max_iterations
        self.newton_start = newton_start
        self.newton_start = newton_start
        
    def find_mode_newton(self, return_full=False):
        """
        Newton search for mode of p(y|f)p(f)
        
        from GP book, algorithm 3.1, added step size
        """
        K = self.gp.K
        
        if self.newton_start is None:
            f = zeros(len(K))
        else:
            f = self.newton_start
            
        if return_full:
            steps = [f]
        
        iteration = 0
        norm_difference = inf
        objective_value = -inf
        
        while iteration < self.newton_max_iterations and norm_difference > self.newton_epsilon:
            # from GP book, algorithm 3.1, added step size
            # scale log_lik_grad_vector and K^-1 f = a
            
            w = -self.gp.likelihood.log_lik_hessian_vector(self.gp.y, f)
            w_sqrt = sqrt(w)
            
            # diag(w_sqrt).dot(K.dot(diag(w_sqrt))) == (K.T*w_sqrt).T*w_sqrt
            L = cholesky(eye(len(K)) + (K.T * w_sqrt).T * w_sqrt)
            b = f * w + self.newton_step * \
                self.gp.likelihood.log_lik_grad_vector(self.gp.y, f)
            
            # a=b-diag(w_sqrt).dot(inv(eye(len(K)) + (K.T*w_sqrt).T*w_sqrt).dot(diag(w_sqrt).dot(K.dot(b))))
            a = (w_sqrt * (K.dot(b)))
            a = solve_triangular(L, a, lower=True)
            a = solve_triangular(L.T, a, lower=False)
            a = w_sqrt * a
            a = b - a
    
            f_new = K.dot(self.newton_step * a)
            
            # convergence stuff and next iteration
            objective_value_new = -0.5 * a.T.dot(f) + \
                                sum(self.gp.likelihood.log_lik_vector(self.gp.y, f))
            norm_difference = norm(f - f_new)
            
            if objective_value_new > objective_value:
                f = f_new
                if return_full:
                    steps.append(f)
            else:
                self.newton_step /= 2
            
            iteration += 1
            objective_value = objective_value_new
            
        self.computed = True
        
        if return_full:
            return f, L, asarray(steps)
        else:
            return f
    
    def get_gaussian(self, f=None, L=None):
        if f is None or L is None:
            f, L, _=self.find_mode_newton(return_full=True)
            
        w = -self.gp.likelihood.log_lik_hessian_vector(self.gp.y, f)
        w_sqrt = sqrt(w)
        K = self.gp.K
            
        # gp book 3.27, matrix inversion lemma on
        # (K^-1 +W)^-1 = K -KW^0.5 B^-1 W^0.5 K
        C = (K.T * w_sqrt).T
        C = solve_triangular(L, C, lower=True)
        C = solve_triangular(L.T, C, lower=False)
        C = (C.T * w_sqrt).T
        C = K.dot(C)
        C = K - C
        
        return Gaussian(f, C, is_cholesky=False)
    
    def predict(self, X_test, f_mode=None):
        """
        Predictions for GP with Laplace approximation.
        
        from GP book, algorithm 3.2,
        
        """
        if f_mode is None:
            f_mode=self.find_mode_newton()
            
        predictions=zeros(len(X_test))    
        
        K = self.gp.K
        K_train_test=self.gp.covariance.compute(self.gp.X, X_test)
        
        w = -self.gp.likelihood.log_lik_hessian_vector(self.gp.y, f_mode)
        w_sqrt = sqrt(w)
        
        # diag(w_sqrt).dot(K.dot(diag(w_sqrt))) == (K.T*w_sqrt).T*w_sqrt
        L = cholesky(eye(len(K)) + (K.T * w_sqrt).T * w_sqrt)

        # iterator for all testing points
        for i in range(len(X_test)):
            k=K_train_test[:,i]
            k_self=self.gp.covariance.compute([X_test[i]], [X_test[i]])[0]
            
            f_mean=k.dot(self.gp.likelihood.log_lik_grad_vector(self.gp.y, f_mode))
            v=solve_triangular(L, w_sqrt*k, lower=True)
            f_var=k_self-v.T.dot(v)
            
            predictions[i]=integrate.quad(lambda x:norm.pdf(x, f_mean, f_var), -inf, inf)[0]
#            # integrate over Gaussian using some crude numerical integration
#            samples=randn(1000)*sqrt(f_var) + f_mean
#            
#            log_liks=self.gp.likelihood.log_lik_vector(1.0, samples)
#            predictions[i]=1.0/len(samples)*GPTools.log_sum_exp(log_liks)
            
            
        return predictions