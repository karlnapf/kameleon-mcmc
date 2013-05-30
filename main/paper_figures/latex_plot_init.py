from matplotlib.pyplot import rc
import matplotlib

rc('text', usetex=True)
rc('text.latex',unicode=True)
#rc('text.latex',preamble='\usepackage{mathpazo}')
rc('font', family='times')
fontsize=8
matplotlib.rcParams.update({'axes.labelsize' : fontsize, \
                            'font.size' : fontsize, \
                            'text.fontsize' : fontsize, \
                            'legend.fontsize': fontsize, \
                            'xtick.labelsize' : fontsize, \
                            'ytick.labelsize' : fontsize})
rc('figure', figsize=(6, 4)) 
