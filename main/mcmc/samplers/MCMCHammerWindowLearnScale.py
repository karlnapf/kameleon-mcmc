from main.mcmc.samplers.MCMCHammerWindow import MCMCHammerWindow
class MCMCHammerWindowLearnScale(MCMCHammerWindow):
    adapt_scale = True
    def __init__(self, distribution, kernel, nu2=0.1, gamma=0.1, window_size=5000, thinning_factor=10):
        MCMCHammerWindow.__init__(self, distribution, kernel, nu2, gamma, window_size, thinning_factor)
        