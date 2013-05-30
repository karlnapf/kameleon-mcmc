from main.distribution.Banana import Banana
from main.distribution.Flower import Flower
from main.distribution.Ring import Ring
def main():
    dist=Ring(dimension=50)
    X=dist.sample(10000).samples
    #print X[:,2:dist.dimension]
    print dist.emp_quantiles(X)
    
    dist2=Banana(dimension=50)
    X2=dist2.sample(10000).samples
    
    print dist2.emp_quantiles(X2)
    
    
    
    
    
main()