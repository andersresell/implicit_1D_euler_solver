from matplotlib import pyplot as plt
import time
from euler_solver_1D import *

#use this to disable asserts
# import __main__
# __main__.__builtins__.__debug__ = False

if __name__ == '__main__':
    

    config = Config(L=3,
                    N=200,
                    CFL=2,
                    stopping_crit=('timesteps', 100),
                    time_discretization='explicit_euler',
                    max_inner_iter=30
                    )
    
    config.add_plotter()
    
    #specifying a shock tube problem with ambient conditions and an overpressure on the left side
    config.specify_riemann_initial_cond(rho_l=1.2,
                                        u_l=0,
                                        p_l=2e5,
                                        rho_r=1.2,
                                        u_r=0,
                                        p_r=1e5)
    
    solver = EulerSolver1D(config)
    
    solver.solve()
    