#!/usr/bin/python
from pickle import load
import sys

assert(len(sys.argv)==2)
filename=str(sys.argv[1])
print "running experiment file", filename

f=open(filename, 'r')
experiment=load(f)
f.close()
experiment.run()