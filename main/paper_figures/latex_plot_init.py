from matplotlib.pyplot import rc
import matplotlib

rc('text', usetex=True)
rc('text.latex',unicode=True)
#rc('text.latex',preamble='\usepackage{mathpazo}')
rc('font', family='Palatino')
matplotlib.rcParams.update({'font.size': 8})
rc('figure', figsize=(6, 4)) 
