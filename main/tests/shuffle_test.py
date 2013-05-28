from numpy.ma.core import arange
from numpy.ma.extras import unique
from numpy.random import randint, randn
import cProfile
import pstats
import time

def main():
    n=1000000
    d=8
    X=randn(n,d)
    
    start_time=time.time()
    times=[0]
    for i in arange(0,n,1000):
        times.append(time.time() - times[-1] - start_time)
        print i, times[-1]
        for _ in arange(1000):
            inds = randint(i+1, size=1000) + 500
            unique_inds=unique(inds)
            
            Z=X[unique_inds]
    
cProfile.run("main()", "profile.tmp")
p = pstats.Stats("profile.tmp")
p.sort_stats("cumulative").print_stats(10)
