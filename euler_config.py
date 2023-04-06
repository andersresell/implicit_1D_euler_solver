import numpy as np
from euler_constants import *

class Config:
    """Contains all settings and some variables"""
    def __init__(self, 
                    L, 
                    N,
                    CFL, 
                    stopping_crit: tuple,
                    flux_type='rusanov', 
                    limiter_type='minmod',
                    left_BC='non_reflective',
                    right_BC='non_reflective',
                    time_discretization = 'explicit_euler',
                    max_inner_iter = 10,
                    inner_iter_epsilon = 1e-6
                    ):
        self.L = L
        self.N = N   
        self.I_DOMAIN = range(2,N+2) 
        self.dx = self.L / self.N  
        self.CFL = CFL
        self.flux_type = flux_type
        self.limiter_type = limiter_type
        self.left_BC = left_BC
        self.right_BC = right_BC
        self.time_discretization = time_discretization
        assert stopping_crit[0] == 'time' or stopping_crit[0] == 'timesteps'
        self.stopping_crit = stopping_crit

        self.t = 0
        self.dt = 1e6
        self.n = 0
        self.stopping_crit_reached = False
        
        self.plot_from_solver = False
        self.initial_cond_specified = False
        
        self.max_inner_iter = max_inner_iter
        self.inner_iter_epsilon = inner_iter_epsilon
        self.i_inner = 0
        
        
    def specify_riemann_initial_cond(self, rho_l, u_l, p_l, rho_r, u_r, p_r):
        self.initial_cond = 'riemann'
        self.rho_l = rho_l
        self.u_l = u_l
        self.p_l = p_l
        self.rho_r = rho_r
        self.u_r = u_r
        self.p_r = p_r
        self.initial_cond_specified = True
        
    def add_plotter(self, animate=True, plot_variable='p', plot_on_inner_iter='False'):
        self.plot_from_solver = True
        self.animate = animate
        self.plot_variable = plot_variable
        self.plot_on_inner_iter = plot_on_inner_iter