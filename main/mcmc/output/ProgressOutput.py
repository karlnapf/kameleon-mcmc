from main.mcmc.output.Output import Output

class ProgressOutput(Output):
    def __init__(self):
        Output.__init__(self)
    
    def update(self, mcmc_chain, step_output):
        it = mcmc_chain.iteration
        if (it % round(mcmc_chain.mcmc_params.num_iterations / 10)) == 0:
            print int(round(float(it) / mcmc_chain.mcmc_params.num_iterations * 100)), \
            "\tpercent done"
    
    def prepare(self):
        pass
