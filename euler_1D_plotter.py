from matplotlib import pyplot as plt
import numpy as np
from euler_config import Config
import euler_utilities as eu
from euler_constants import *

class Plotter1D:
    def __init__(self, config: Config):
        dx = config.dx
        L = config.L
        N = config.N
        self.x = np.linspace(start=dx/2, stop=L-dx/2, num=N)
        self.val = np.zeros_like(self.x)
        plt.figure()
    def update(self, U, config: Config):
        assert U.shape[0] == config.N+N_GHOST and U.shape[1] == N_VAR
        
        if config.animate or config.stopping_crit_reached:
            plt.gca()
            
            if config.plot_variable == 'p':
                for i in range(2,config.N+2):
                    self.val[i] = eu.calc_pressure(U[i])
            else: raise ValueError(f"Invalid variable to plot: {config.plot_variable}")
            
            plt.plot(self.x, self.val)
            plt.pause(0.000001)
            