from main.distribution.Flower import Flower
from main.tools.Visualise import Visualise

class Ring(Flower):
    def __init__(self, variance=0.05, radius=3.5, dimension=2):
        Flower.__init__(self, 0, 1, variance, radius, dimension)

if __name__ == '__main__':
    Visualise.visualise_distribution(Ring())
