#from classes.distribution.Gaussian import Gaussian
#from kernel.Kernel import Kernel
#from numpy.ma.core import sqrt, array
#
#class GFunction(object):
#    def __init__(self, kernel, distribution, y=array([[-5, -1]]), \
#                 epsilon=sqrt(200), ell=15):
#        self.kernel = kernel
#        self.distribution = distribution
#        self.y=y
#        self.epsilon=epsilon
#        self.ell=ell
#        
#    def evaluate(self, n=200):
#        Z = self.distribution.sample(n)
#        
#        # plot points
#    #    figure(figsize=(15,10))
#    #    plot(Z[:, 0], Z[:, 1], '*')
#    #    show()
#        
#        # evaluate and center kernel and scale
#        K = self.kernel.kernel(Z, None)
#        K = Kernel.center_kernel_matrix(K)
#        K *= self.epsilon ** 2
#        
#    #    # plot kernel matrix
#    #    figure(figsize=(15,10))
#    #    imshow(Kc, interpolation='nearest')
#    #    show()
#        
#        # sample beta and fix current point y
#    #    beta = sample_gaussian(L, 1, is_cholesky=True)
#        gaussian=Gaussian(self, mu=array([0, 0]), Sigma=K,
#                          is_cholesky=False, ell=self.ell)
#        beta = gaussian.sample()
#        
#        # precompute constants
#        constants = mcmc_hammer_proposal_log_pdf_constants(y, (Z, eta, gamma, kernel_gradient))
#    
#        # where to evaluate G?
#        GXs = linspace(-15, 15, 70)
#        GYs = linspace(-5, 10, 40)
#        G = zeros((len(GYs), len(GXs)))
#        G_gradient_norm = zeros((len(GYs), len(GXs)))
#        P = zeros((len(GYs), len(GXs)))
#    
#        # for plotting the gradient field, each U and V are one dimension of gradient
#        GXs2 = linspace(GXs.min(), GXs.max(), 30)
#        GYs2 = linspace(GYs.min(), GYs.max(), 20)
#        X, Y = meshgrid(GXs2, GYs2)
#        U = zeros(shape(X))
#        V = zeros(shape(Y))
#    
#        # evaluate g at a set of points in GXy and GYs
#        for i in range(len(GXs)):
#            print i, "/", len(GXs)
#            for j in range(len(GYs)):
#                x = array([[GXs[i], GYs[j]]])
#                G[j, i] = compute_g(x, y, Z, beta, kernel)
#                grad = compute_g_gradient(x, y, Z, beta, kernel_gradient)
#                G_gradient_norm[j, i] = norm(grad)
#                P[j, i] = mv_log_normal_pdf(x, constants[0], constants[1], is_cholesky=True)
#    
#        # gradient at lower resolution
#        for i in range(len(GXs2)):
#            print i, "/", len(GXs2)
#            for j in range(len(GYs2)):
#                x = array([[GXs2[i], GYs2[j]]])
#                grad = compute_g_gradient(x, y, Z, beta, kernel_gradient)
#                U[j, i] = -grad[0, 0]
#                V[j, i] = -grad[0, 1]
#    
#        # plot g and Z points and y
#        figure(figsize=(15, 10))
#        suptitle('g function')
#        plot_array(GXs, GYs, G)
#        hold(True)
#        plot_data(Z, y)
#        hold(False)
#        savefig("g_gunction.png")
#        
#        figure(figsize=(15, 10))
#        suptitle("g gradient norm")
#        plot_array(GXs, GYs, G_gradient_norm)
#        hold(True)
#        plot_data(Z, y)
#        hold(False)
#        savefig("g_gradient_norm.png")
#        
#        figure(figsize=(15, 10))
#        suptitle("g gradient field")
#        plot_array(GXs, GYs, G_gradient_norm)
#        hold(True)
#        plot_data(Z, y)
#        quiver(X, Y, U, V, color='y', scale=G_gradient_norm.max() * 15)
#        hold(False)
#        savefig("g_gradient_field.png")
#        
#        figure(figsize=(15, 10))
#        suptitle("Proposal density")
#        plot_array(GXs, GYs, exp(P))
#        hold(True)
#        plot_data(Z, y)
#        hold(False)
#        savefig("proposal_pdf.png")
#        
#        show()
