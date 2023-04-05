import numpy as np
from euler_constants import *
import euler_utilities as eu
from euler_config import Config
from euler_1D_plotter import Plotter1D
#import numba as nb

class EulerSolver1D:
    """An implicit solver for the 1D Euler equations"""
    
    def __init__(self, config: Config):
        
        self.config = config
        self.L = config.L
        self.N = config.N
        self.I_DOMAIN = config.I_DOMAIN
        self.dx = self.L/self.N
        
        self.U_old = np.zeros((self.N + N_GHOST, N_VAR))
        self.U = np.zeros_like(self.U_old)
        self.R = np.zeros((self.N, N_VAR))
        self.dR = np.zeros((self.N * N_VAR, self.N * N_VAR))
        
        assert config.initial_cond_specified == True
        if config.initial_cond == 'riemann':
            eu.set_initial_cond_riemann(self.U_old, self.config)
        else:
            raise ValueError(f"Invalid initial condition type: {config.initial_cond}")
        
        
        if config.flux_type == 'rusanov':
            self.calc_numerical_flux = eu.calc_rusanov_flux
        else:
            raise ValueError(f"Invalid flux type: {config.limiter_type}")
        
        if config.limiter_type == 'minmod':
            self.limiter = eu.minmod
        else: raise ValueError(f"Invalid limiter type: {config.limiter_type}")
        
        if config.plot_from_solver:
            self.plotter = Plotter1D(config)
            
    def solve(self):
        self.config.t = 0
        self.config.n = 0
        
        while True:
            print('n =', self.config.n)
            # print(self.U_old)
            # print(self.config.n)
            # print(self.config.dt)
            
            if self.config.plot_from_solver:
                self.plotter.update(self.U_old, self.config)
            
            
            eu.calc_dt(self.config, self.U_old)
            
            #compute time step
            self.step()
            
            self.config.t += self.config.dt
            self.config.n += 1
            
            
            if eu.stopping_crit_reached(self.config):
                print('stopping criterion reached')
                break
            
    def step(self):
        if self.config.time_discretization == 'explicit_euler':
            self.explicit_euler_step()
        elif self.config.time_discretization == 'TVD_RK3':
            self.TVD_RK3_step()
        else: raise ValueError(f"Invalid time discretization: {self.config.time_discretization}")
            
    def explicit_euler_step(self):
        eu.set_ghost_cells(self.config, self.U_old)
        self.evaluate_R(self.U_old)
        
        self.U[self.I_DOMAIN] = self.U_old[self.I_DOMAIN] + self.config.dt * self.R
        self.U_old = self.U

    def TVD_RK3_step(self):
        dt = self.config.dt
        eu.set_ghost_cells(self.config, self.U_old)
        self.evaluate_R(self.U_old)
        self.U[self.I_DOMAIN] = self.U_old[self.I_DOMAIN] + dt * self.R
        
        eu.set_ghost_cells(self.config, self.U)
        self.evaluate_R(self.U)
        self.U[self.I_DOMAIN] = 3/4 * self.U_old[self.I_DOMAIN] + 1/4 * self.U[self.I_DOMAIN] + 1/4 * dt * self.R

        eu.set_ghost_cells(self.config, self.U)
        self.evaluate_R(self.U)
        self.U[self.I_DOMAIN] = 1/3 * self.U_old[self.I_DOMAIN] + 2/3 * self.U[self.I_DOMAIN] + 2/3 * dt * self.R 
        self.U_old = self.U
    

    def evaluate_R(self, U):
        
        self.R *= 0

        #loop over all faces
        for i in range(0, self.N+1):
            (U_L, U_R) = eu.calc_extrapolated_values(i+1, U, self.limiter)
            #print('f\n',self.calc_numerical_flux(U_L, U_R))
            F_face_res = self.calc_numerical_flux(U_L, U_R) / self.dx
            
            if i == 0: #first face
                self.R[0] += F_face_res
            elif i == self.N: #last face
                self.R[self.N-1] -= F_face_res
                
            else:
                self.R[i-1] -= F_face_res
                self.R[i] += F_face_res
            
                    

    
    
    
    
    
    

    
    