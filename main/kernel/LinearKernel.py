from main.kernel.Kernel import Kernel

class LinearKernel(Kernel):
    def __init__(self):
        Kernel.__init__()
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "" + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y):
        """
        Computes the linear kernel k(x,y)=x^T y for the given data
        X - samples on right hand side
        Y - samples on left hand side, can be None in which case its replaced by X
        """
        if Y is None:
            Y = X
        
        return X.dot(Y.T)

    def gradient(self, x, Y, args_euqal=False):
        """
        Computes the linear kernel k(x,y)=x^T y for the given data
        x - single sample on right hand side
        Y - samples on left hand side
        """
        return Y
