from main.distribution.Gaussian import Gaussian
from main.distribution.Ring import Ring
from main.kernel.GaussianKernel import GaussianKernel
from main.kernel.Kernel import Kernel
from main.mcmc.samplers.MCMCHammer import MCMCHammer
from main.tools.Visualise import Visualise
from matplotlib.pyplot import suptitle, hold, savefig, quiver, show, figure
from numpy.core.function_base import linspace
from numpy.core.numeric import zeros
from numpy.lib.function_base import meshgrid
from numpy.ma.core import array, shape, exp

class GFunction(object):
    def __init__(self, distribution, gaussian_width=1, eta=0.1, gamma=0.1, ell=15):
        self.kernel = GaussianKernel(gaussian_width)
        self.distribution = distribution
        self.eta = eta
        self.gamma = gamma
        self.ell = ell
        
    def compute(self, x, y, Z, beta):
        """
        Given two points x and y, a set of samples Z, and a vector beta,
        and a kernel function, this computes the g function
        g(x,beta,Z)=||k(x,.)-f|| for f=k(.,y)+sum_i beta_i*k(.,z_i)
                   =k(x,x) -2k(x,y) -2sum_i beta_i*k(x,z_i) +C
        Constant C is not computed
        """
        first = self.kernel.kernel(x, x)
        second = -2 * self.kernel.kernel(x, y)
        third = -2 * self.kernel.kernel(x, Z).dot(beta.T)
        return first + second + third

    def compute_gradient(self, x, y, Z, beta):
        """
        Given two points x and y, a set of samples Z, and a vector beta,
        and a kernel gradient, this computes the g function's gradient
        \nabla_x g(x,beta,Z)=\nabla_x k(x,x) -2k(x,y) -2sum_i beta_i*k(x,z_i)
        """
        first = self.kernel.gradient(x, x)
        second = -2 * self.kernel.gradient(x, y)
        
        # compute sum_i beta_i \nabla_x k(x,z_i) and beta is a row vector
        gradients = self.kernel.gradient(x, Z)
        third = -2 * beta.dot(gradients)
        
        return first + second + third


    def plot(self, y=array([[-2, -2]]), n=200):
        Z = self.distribution.sample(n)
        
        # evaluate and center kernel and scale
        K = self.kernel.kernel(Z, None)
        K = Kernel.center_kernel_matrix(K)
        
        # sample beta and fix current point y
        gaussian = Gaussian(mu=zeros(n), Sigma=K, is_cholesky=False, \
                          ell=self.ell)
        beta = gaussian.sample()
        
        # precompute constants
        mcmc_hammer = MCMCHammer(self.distribution, self.kernel, Z, self.eta, self.gamma)
        mu, L_R = mcmc_hammer.compute_constants(y)
    
        # where to evaluate G?
        GXs = linspace(-15, 15, 70)
        GYs = linspace(-5, 10, 40)
        G = zeros((len(GYs), len(GXs)))
        P = zeros((len(GYs), len(GXs)))
    
        # for plotting the gradient field, each U and V are one dimension of gradient
        GXs2 = linspace(GXs.min(), GXs.max(), 30)
        GYs2 = linspace(GYs.min(), GYs.max(), 20)
        X, Y = meshgrid(GXs2, GYs2)
        U = zeros(shape(X))
        V = zeros(shape(Y))
    
        # evaluate g at a set of points in GXy and GYs
        figure()
        gaussian = Gaussian(mu, L_R, is_cholesky=True)
        print L_R.T.dot(L_R)
        Visualise.visualise_distribution(gaussian)
        figure()
        for i in range(len(GXs)):
            print i, "/", len(GXs)
            for j in range(len(GYs)):
                x = array([[GXs[i], GYs[j]]])
                G[j, i] = self.compute(x, y, Z, beta)
                P[j, i] = gaussian.log_pdf(x)
    
        # gradient at lower resolution
        for i in range(len(GXs2)):
            print i, "/", len(GXs2)
            for j in range(len(GYs2)):
                x = array([[GXs2[i], GYs2[j]]])
                G_grad = self.compute_gradient(x, y, Z, beta)
                U[j, i] = -G_grad[0, 0]
                V[j, i] = -G_grad[0, 1]
    
        # plot g and Z points and y
        figure(figsize=(15, 10))
        suptitle("g function with gradient")
        Visualise.plot_array(GXs, GYs, G)
        hold(True)
        Visualise.plot_data(Z, y)
        quiver(X, Y, U, V, color='y', scale=G.max() * 15)
        hold(False)
        savefig("g_function_with_gradient.png")
        
        figure(figsize=(15, 10))
        suptitle("Proposal density")
        Visualise.plot_array(GXs, GYs, exp(P))
        hold(True)
        Visualise.plot_data(Z, y)
        hold(False)
        savefig("proposal_pdf.png")
        
        show()

if __name__ == '__main__':
    distribution = Ring()
    g_func = GFunction(distribution)
    
    y = array([[2, -3 ]])
    g_func.plot(y, n=1000)
