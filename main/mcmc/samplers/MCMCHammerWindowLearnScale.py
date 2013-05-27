from main.mcmc.samplers.MCMCHammerWindow import MCMCHammerWindow
class MCMCHammerWindowLearnScale(MCMCHammerWindow):
    adapt_scale = True
    def __init__(self, distribution, kernel, nu2=0.1, gamma=0.1, sample_discard=500, num_samples_Z=1000):
        MCMCHammerWindow.__init__(self, distribution, kernel, nu2, gamma, sample_discard, num_samples_Z)
        