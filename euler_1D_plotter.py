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
        
        self.first_update = True
        
    def update(self, U, config: Config):
        assert U.shape[0] == config.N+N_GHOST and U.shape[1] == N_VAR   
        if config.animate or config.stopping_crit_reached:
            plt.cla()
            
            if config.plot_variable == 'p':
                for i in range(0, config.N):
                    self.val[i] = eu.calc_pressure(U[i+2])
            else: raise ValueError(f"Invalid variable to plot: {config.plot_variable}")
            
            plt.plot(self.x, self.val,'k.')
            
            if self.first_update:
                self.bottom, self.top = plt.ylim()
            plt.ylim(self.bottom, self.top)
            
            plt.pause(0.000001)
            
            self.first_update = False
            