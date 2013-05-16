from main.distribution.Flower import Flower
from main.tools.Visualise import Visualise
from numpy.ma.core import sqrt

class Ring(Flower):
    def __init__(self, variance=0.05, radius=3.5, dimension=2):
        Flower.__init__(self, 0, 1, variance, radius, dimension)

    def get_plotting_bounds(self):
        if self.dimension==2:
            value=self.radius + 3*sqrt(self.variance)
            return [(-value, value) for _ in range(2)]
        else:
            return Flower.get_plotting_bounds(self)

if __name__ == '__main__':
    Visualise.visualise_distribution(Ring())

