from matplotlib import pyplot as plt
import time
from euler_solver_1D import *

#use this to disable asserts
# import __main__
# __main__.__builtins__.__debug__ = False

if __name__ == '__main__':
    
    times = []
    n_counts = 5
    for i in range(0,n_counts):
        config = Config(L=3,
                        N=300,
                        CFL=0.3,
                        stopping_crit=('timesteps', 200),
                        time_discretization='TVD_RK3'
                        )
        
        #config.add_plotter()
        
        #specifying a shock tube problem with ambient conditions and an overpressure on the left side
        config.specify_riemann_initial_cond(rho_l=1.2,
                                            u_l=0,
                                            p_l=2e5,
                                            rho_r=1.2,
                                            u_r=0,
                                            p_r=1e5)
        
        solver = EulerSolver1D(config)
        
        start_time = time.time()
        solver.solve()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print('elapsed time: ', elapsed_time, 'seconds')
        times.append(elapsed_time)
    print(times)
    plt.show()