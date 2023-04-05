import numpy as np
import numba as nb

@nb.vectorize
def fill_matrix(mat, N, M):
    for i in range(0,N):
        for j in range(0,M):
            mat[i][j] = j-i

@nb.jit
def add(A,B):
    N,M = A.shape
    C = np.zeros_like(A)
    for i in range(0,N):
        for j in range(0,M):
            C[i][j] = A[i][j]+B[i][j]
    return C

@nb.vectorize
def add_np(A,B):
    C = A+B
    return C
N = 1000
M = 20000
A = 0.5* np.ones((N,M))
B = 0.3*np.ones((N,M))
import time

t0 = time.time()

C = add(A,B)

t1 = time.time()

print('time:',t1-t0)



t0 = time.time()

C = add_np(A,B)

t1 = time.time()

print('time:',t1-t0)