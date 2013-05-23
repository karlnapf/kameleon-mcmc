from main.distribution.Distribution import Distribution
from main.distribution.Flower import Flower
from numpy.ma.core import sqrt

class Ring(Flower):
    def __init__(self, variance=0.05, radius=3.5, dimension=2):
        Flower.__init__(self, 0, 1, variance, radius, dimension)

    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "variance="+ str(self.variance)
        s += ", radius="+ str(self.radius)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s

    def get_plotting_bounds(self):
        if self.dimension==2:
            value=self.radius + 3*sqrt(self.variance)
            return [(-value, value) for _ in range(2)]
        else:
            return Flower.get_plotting_bounds(self)

#if __name__ == '__main__':
#    ring_instance=Ring()
#    X=ring_instance.sample(10000).samples
#    print ring_instance.emp_quantiles(X)
#    Visualise.visualise_distribution(Ring())
