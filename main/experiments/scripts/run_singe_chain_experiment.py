#!/usr/bin/python

# add path
import os
import sys
to_add=os.sep.join(os.path.abspath(os.path.dirname(sys.argv[0])).split(os.sep)[0:-3])
sys.path.append(to_add)

from pickle import load

assert(len(sys.argv)==2)
filename=str(sys.argv[1])
print "running experiment file", filename

f=open(filename, 'r')
experiment=load(f)
f.close()
experiment.run()