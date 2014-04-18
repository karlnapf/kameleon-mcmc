"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from matplotlib.pyplot import rc
import matplotlib

rc('text', usetex=True)
rc('text.latex',unicode=True)
rc('text.latex',preamble='\usepackage{amsfonts}')
rc('font', family='times')
fontsize=8
matplotlib.rcParams.update({'axes.labelsize' : fontsize, \
                            'font.size' : fontsize, \
                            'text.fontsize' : fontsize, \
                            'legend.fontsize': fontsize, \
                            'xtick.labelsize' : fontsize, \
                            'ytick.labelsize' : fontsize})
rc('figure', figsize=(6, 4)) 
