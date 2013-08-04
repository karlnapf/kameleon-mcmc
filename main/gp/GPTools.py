from numpy.lib.function_base import delete
from numpy.ma.core import log, exp
class GPTools(object):
    @staticmethod
    def log_sum_exp(X):
        """
        Computes log sum_i exp(X_i).
        Useful if you want to solve log \int f(x)p(x) dx
        where you have samples from p(x) and can compute log f(x)
        """
        # extract minimum
        X0=X.min()
        X_without_X0=delete(X,X.argmin())
        
        return X0+log(1+sum(exp(X_without_X0-X0)))
    
    @staticmethod
    def log_mean_exp(X):
        """
        Computes log 1/n sum_i exp(X_i).
        Useful if you want to solve log \int f(x)p(x) dx
        where you have samples from p(x) and can compute log f(x)
        """
        
        return GPTools.log_sum_exp(X)-log(len(X))