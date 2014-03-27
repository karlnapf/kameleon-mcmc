from matplotlib.pyplot import axis, savefig
from numpy import linspace

from main.distribution.Banana import Banana
from main.distribution.Flower import Flower
from main.distribution.Ring import Ring
from main.tools.Visualise import Visualise


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
