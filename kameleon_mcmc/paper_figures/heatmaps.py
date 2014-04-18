from matplotlib.pyplot import axis, savefig
from numpy import linspace

from kameleon_mcmc.distribution.Banana import Banana
from kameleon_mcmc.distribution.Flower import Flower
from kameleon_mcmc.distribution.Ring import Ring
from kameleon_mcmc.tools.Visualise import Visualise


if __name__ == '__main__':
    distributions = [Ring(), Banana(), Flower()]
    for d in distributions:
        Xs, Ys = d.get_plotting_bounds()
        resolution = 250
        Xs = linspace(Xs[0], Xs[1], resolution)
        Ys = linspace(Ys[0], Ys[1], resolution)
        
        Visualise.visualise_distribution(d, Xs=Xs, Ys=Ys)
        axis("Off")
        savefig("heatmap_" + d.__class__.__name__ + ".eps", bbox_inches='tight')
