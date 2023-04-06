import numpy as np
from matplotlib import pyplot as plt
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
            
        if self.config.plot_from_solver:
            self.plotter.update(self.U_old, self.config)
            plt.show()    
                
    def step(self):
        time_disc = self.config.time_discretization
        if time_disc == 'explicit_euler':
            self.explicit_euler_step()
        elif time_disc == 'TVD_RK3':
            self.TVD_RK3_step()
        elif time_disc == 'implicit_euler' or time_disc == 'backward_differencing_2nd':
            self.implicit_step()
        else: raise ValueError(f"Invalid time discretization: {self.config.time_discretization}")
            
    def explicit_euler_step(self):
        eu.set_ghost_cells(self.config, self.U_old)
        self.evaluate_residual(self.U_old)
        
        self.U[self.I_DOMAIN] = self.U_old[self.I_DOMAIN] + self.config.dt * self.R
        self.U_old = self.U

    def TVD_RK3_step(self):
        dt = self.config.dt
        eu.set_ghost_cells(self.config, self.U_old)
        self.evaluate_residual(self.U_old)
        self.U[self.I_DOMAIN] = self.U_old[self.I_DOMAIN] + dt * self.R
        
        eu.set_ghost_cells(self.config, self.U)
        self.evaluate_residual(self.U)
        self.U[self.I_DOMAIN] = 3/4 * self.U_old[self.I_DOMAIN] + 1/4 * self.U[self.I_DOMAIN] + 1/4 * dt * self.R

        eu.set_ghost_cells(self.config, self.U)
        self.evaluate_residual(self.U)
        self.U[self.I_DOMAIN] = 1/3 * self.U_old[self.I_DOMAIN] + 2/3 * self.U[self.I_DOMAIN] + 2/3 * dt * self.R 
        self.U_old = self.U
    
    def implicit_step(self):
        self.config.i_inner = 0
        dt = self.config.dt
        
        eu.set_ghost_cells(self.config, self.U_old)
        self.U[self.I_DOMAIN] = self.U_old[self.I_DOMAIN]
        #print('U_old',self.U_old)
        while True:
            self.config.i_inner += 1
            
            eu.set_ghost_cells(self.config, self.U)
            self.evaluate_residual(self.U)
            self.evaluate_residual_jacobian(self.U)

            #print('dR',self.dR)
        
            LHS = np.identity(N_VAR * self.N) / dt - self.dR
            if self.config.time_discretization == 'implicit_euler':
                RHS = (-(self.U[self.I_DOMAIN] - self.U_old[self.I_DOMAIN])/dt + self.R).flatten()
            else: 
                raise ValueError(f"Invalid time discretization: {self.config.time_discretization}")

            delta_U = np.linalg.solve(LHS, RHS).reshape((self.N, N_VAR))
            self.U[self.I_DOMAIN] += delta_U
            
            self.plotter.update(self.U, self.config)
            
            L2_density_res = eu.L2_norm_density(delta_U, self.config)
            #print('delta U',delta_U)
            print('dt',dt)
            #print('U_k',self.U)
            print('res',L2_density_res)
            
            if eu.stop_inner_iter(L2_density_res, self.config):
                break
        #print('U_old',self.U_old)
        #print('delta',self.U - self.U_old)
        self.U_old[self.I_DOMAIN] = self.U[self.I_DOMAIN]
        #print('U_old',self.U_old)
            
    def evaluate_residual(self, U):
        
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
            
    def evaluate_residual_jacobian(self, U):
        assert U.shape == (self.N + N_GHOST, N_VAR)
        self.dR *= 0
        
        IND = lambda I,J: (slice(N_VAR*I, N_VAR*(I+1)), slice(N_VAR*J, N_VAR*(J+1))) #to access a (N_VAR X N_VAR) block matrix
        
        for i in range(0,self.N):
            U_im= U[i+1]
            U_i = U[i+2]
            U_ip = U[i+3]
            
            #approximation using first order rusanov fluxes 
            spec_rad_plus = max(eu.calc_spectral_radii(U_ip), eu.calc_spectral_radii(U_i))
            spec_rad_minus = max(eu.calc_spectral_radii(U_i), eu.calc_spectral_radii(U_im))
            
        
            #print('submat:\n',0.5*(-eu.calc_flux_jacobian(U_ip) + spec_rad_plus * IDENTITY_N_VAR))
            if i == 0:
                self.dR[IND(i,i+1)] = 0.5*(-eu.calc_flux_jacobian(U_ip) + spec_rad_plus * IDENTITY_N_VAR)
    
            elif i == self.N-1:
                self.dR[IND(i,i-1)] = 0.5 * (eu.calc_flux_jacobian(U_im) + spec_rad_minus* IDENTITY_N_VAR)
            else:
                self.dR[IND(i,i-1)] = 0.5 * (eu.calc_flux_jacobian(U_im) + spec_rad_minus * IDENTITY_N_VAR)

                #print('submat',0.5*(-eu.calc_flux_jacobian(U_ip) + spec_rad_plus * IDENTITY_N_VAR))
                self.dR[IND(i,i+1)] = 0.5*(-eu.calc_flux_jacobian(U_ip) + spec_rad_plus * IDENTITY_N_VAR)
                
            self.dR[IND(i,i)] = -0.5 * (spec_rad_plus + spec_rad_minus) / self.dx * IDENTITY_N_VAR
    
    
    
    
    

    
    