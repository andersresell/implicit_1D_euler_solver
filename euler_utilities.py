
import numpy as np
from euler_constants import *
from euler_config import Config
#import numba as nb

def conservative2primitive(U):
    assert U.shape == (N_VAR,)
    return np.array([U[0],
                     U[1] / U[0],
                     calc_pressure(U)])
    
def primitive2conservative(V):
    assert V.shape == (N_VAR,)
    return np.array([V[0],
                    V[0] * V[1],
                    V[2] * GAMMA_MINUS_ONE_INV + 0.5 * V[0] * V[1]**2])
    
    
def calc_rusanov_flux(U_L, U_R):
    assert U_L.shape == (N_VAR,) and U_R.shape == (N_VAR,)
    return 0.5 * (calc_flux(U_L) + calc_flux(U_R) - \
        max(calc_spectral_radii(U_L), calc_spectral_radii(U_R)) * (U_R - U_L))

def calc_pressure(U):
    assert U.shape == (N_VAR,)
    p = (GAMMA - 1) * (U[2] - 0.5 * U[1]**2 / U[0])
    assert p > 0
    return p

def calc_sound_speed(U):
    assert U.shape == (N_VAR,)
    c = (GAMMA * calc_pressure(U) / U[0])**0.5  
    assert c > 0
    return c

def calc_spectral_radii(U):
    assert U.shape == (N_VAR,)
    return abs(U[1]/U[0]) + calc_sound_speed(U)    

def calc_flux(U):
    assert U.shape == (N_VAR,)
    return np.array([U[1],
                    0.5*(3-GAMMA)*U[1]**2/U[0] + (GAMMA-1)*U[2],
                    GAMMA*U[1]/U[0]*U[2] - 0.5*(GAMMA-1)*U[1]**3/U[0]**2])
    
    # p = calc_pressure(U)
    # F1 = np.array([U[1],
    #                     U[1]**2 / U[0] + p,
    #                     U[1] / U[0] * (U[2] + p)])
    # F2 = np.array([U[1],
    #                0.5*(3-GAMMA)*U[1]**2/U[0] + (GAMMA-1)*U[2],
    #                GAMMA*U[1]/U[0]*U[2] - 0.5*(GAMMA-1)*U[1]**3/U[0]**2])
    
    # equal_elem = F1 == F2
    # print('F1',F1,'F2',F2)
    # assert equal_elem.all()
    # return F2

def minmod(a,b):
    assert a.shape == (N_VAR,) and b.shape == (N_VAR,)
    
    # #chat gpt solution, check it!
    # signs = np.sign(a) * np.sign(b)
    # mins = np.minimum(np.abs(a), np.abs(b))
    # result = np.where(signs > 0, signs * mins, 0)
    # return result
    sign = np.sign(a)
    return sign * np.maximum(np.zeros((N_VAR,)), np.minimum(np.abs(a), sign * b)) 

def calc_extrapolated_values(i, U, limiter):
    """returns U_L and U_R at face i+1/2. Extrapolation is done with primite variables"""
    V_im = conservative2primitive(U[i-1])
    V_i = conservative2primitive(U[i])
    V_ip = conservative2primitive(U[i+1])
    V_ipp = conservative2primitive(U[i+2])
    V_L = V_i + 0.5 * limiter(V_i - V_im, V_ip - V_i)
    V_R = V_ip - 0.5 * limiter(V_ip - V_i, V_ipp - V_ip)
    
    assert V_L.shape == (N_VAR,) and V_R.shape == (N_VAR,)
    return (primitive2conservative(V_L), primitive2conservative(V_R))
    # U_L = U[i] + 0.5 * limiter(U[i] - U[i-1], U[i+1] - U[i])
    # U_R = U[i+1] - 0.5 * limiter(U[i+1] - U[i], U[i+2] - U[i+1])
    # assert U_L.shape == (N_VAR,) and U_R.shape == (N_VAR,)
    # return (U_L, U_R)

def set_ghost_cells(config: Config, U):
    N = config.N
    assert U.shape ==(N+N_GHOST,N_VAR)
    
    if config.left_BC == 'non_reflective':
        U[0] = U[3]
        U[1] = U[2]
    else:
        raise ValueError(f"Invalid BC type: {config.left_BC}")
    
    if config.right_BC == 'non_reflective':
        U[N+2] = U[N+1]
        U[N+3] = U[N]
    else:
        raise ValueError(f"Invalid BC type: {config.left_BC}")
    
def calc_dt(config: Config, U):
    assert U.shape == (config.N+N_GHOST, N_VAR)
    dt = 1e6
    for i in range(2, config.N+2):
        dt = min(dt, config.dx * config.CFL / calc_spectral_radii(U[i]))
    config.dt = dt
   
    
def stopping_crit_reached(config: Config):
    if config.stopping_crit[0] == 'time':
        if config.t >= config.stopping_crit[1]:
            config.stopping_crit_reached = True
            return True
    elif config.stopping_crit[0] == 'timesteps':
        if config.n >= config.stopping_crit[1]:
            config.stopping_crit_reached = True
            return True
    else:
        raise ValueError(f"Invalid stopping criterion: {config.stopping_crit[0]}")
    return False

def stop_inner_iter(residual ,config: Config):
    if residual < config.inner_iter_epsilon:
        print('Inner iterations converged in',config.i_inner, 'iterations. Residual =', residual)
        return True
    elif config.i_inner >= config.max_inner_iter:
        print('Inner iterations did not converge in',config.max_inner_iter, 'iterations')
        return True
    return False

def set_initial_cond_riemann(U, config: Config):
    assert U.shape == (config.N+N_GHOST, N_VAR) and config.initial_cond == 'riemann'
    
    V_L = np.array([config.rho_l, config.u_l, config.p_l])
    V_R = np.array([config.rho_r, config.u_r, config.p_r])
    
    U_L = primitive2conservative(V_L)
    U_R = primitive2conservative(V_R)
    
    for i in range(2, config.N+2):
        if i <= int(config.N/2)+1:
            U[i] = U_L
        else:
            U[i] = U_R

def calc_flux_jacobian(U):
    assert U.shape == (N_VAR,)
    U1_U0 = U[1]/U[0]
    U2_U0 = U[2]/U[0]
    A = np.array([[0,                                      1,                                    0],
                    [-0.5*(3-GAMMA)*U1_U0**2,                 (3-GAMMA)*U1_U0,                      GAMMA-1],
                    [-GAMMA*U1_U0*U2_U0 + (GAMMA-1)*U1_U0**3, GAMMA*U2_U0 - 1.5*(GAMMA-1)*U1_U0**2, GAMMA*U1_U0]])  
    
    #remove when verified
    assert np.isclose(np.matmul(A,U), calc_flux(U)).all()
    return A

def L2_norm_density(U, config: Config):
    assert U.shape == (config.N, N_VAR)
    return (np.dot(U[:,0].T, U[:,0]) * config.dx)**0.5