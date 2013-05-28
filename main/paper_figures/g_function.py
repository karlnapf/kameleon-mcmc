from main.distribution.Banana import Banana
from main.distribution.Ring import Ring
from main.paper_figures.GFunction import GFunction
from matplotlib.pyplot import savefig, legend, show, figure
from numpy.ma.core import array
from random import seed
import latex_plot_init

# global variables that are used by both functions
distributions = [Ring(), Banana()]
sigmas = [.5, 3]
ns=[200, 200]
g_functions = [GFunction(distributions[i], n=ns[i], gaussian_width=sigmas[i], \
                         nu2=0.1, gamma=0.1, ell=15) for i in range(len(distributions))]
ys= [array([2, -3 ]) , array([2, -3 ]) ]

def plot_g_functions(seed_init):
    
    for i in range(len(distributions)):
        seed(seed_init)
        figure()
        
        g_functions[i].plot(ys[i], plot_gradient=True, plot_data=True)
        legend(["Samples $\{z_i\}_{i=1}^{" + str(ns[i]) + "}$", "Current position $y$"], numpoints=1, loc="upper center")
        savefig("g_gunction_" + distributions[i].__class__.__name__ + ".pdf")
        
    show()
    
def plot_proposals(seed_init):
    for i in range(len(distributions)):
        seed(seed_init)
        figure()
        
        ys=distributions[i].sample(2).samples
        
        g_functions[i].plot_proposal(ys)
        show()
        
        

if __name__ == '__main__':
#    plot_g_functions(1)
    plot_proposals(1)
    
    
