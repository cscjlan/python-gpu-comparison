import numpy as np
import numba as nb
from numba import cuda
import matplotlib.pyplot as plt
import time
dtype = np.float64
jit_parallel = nb.njit(nogil=True,parallel=True)


dtype = np.float64

@jit_parallel
def compute_macro_vars(f,nodetype,rho,u):
    ny,nx = nodetype.shape
    for i in nb.prange(ny):
        for j in range(nx):
            if nodetype[i,j]<=0:
                rho[i,j]=f[i,j,0]+f[i,j,1]+f[i,j,2]+f[i,j,3]+f[i,j,4]+f[i,j,5]+f[i,j,6]+f[i,j,7]+f[i,j,8]
                fdotex= f[i,j,1] + f[i,j,3] + f[i,j,4] - f[i,j,5] - f[i,j,7] - f[i,j,8]
                fdotey= f[i,j,2] + f[i,j,3] - f[i,j,4] - f[i,j,6] - f[i,j,7] + f[i,j,8]
                u[0,i,j]= fdotex/rho[i,j]
                u[1,i,j]= fdotey/rho[i,j]                
            else:
                rho[i,j]=0
                u[:,i,j]=0

@cuda.jit
def compute_macro_vars_gpu(f, nodetype, rho, u):
    i, j = cuda.grid(2)
    ny, nx = nodetype.shape
    if i < ny and j < nx and nodetype[i, j] <= 0:
        rho[i, j] = f[i, j, 0] + f[i, j, 1] + f[i, j, 2] + f[i, j, 3] + f[i, j, 4] + f[i, j, 5] + f[i, j, 6] + f[i, j, 7] + f[i, j, 8] 
        fdotex = f[i, j, 1] + f[i, j, 3] + f[i, j, 4] - f[i, j, 5] - f[i, j, 7] - f[i, j, 8]
        fdotey = f[i, j, 2] + f[i, j, 3] - f[i, j, 4] - f[i, j, 6] - f[i, j, 7] + f[i, j, 8]
        u[0, i, j] = fdotex / rho[i, j]
        u[1, i, j] = fdotey / rho[i, j]
    elif i < ny and j < nx:
        rho[i, j] = 0
        u[0, i, j] = 0
        u[1, i, j] = 0


def test_macro():
    nx = 10000
    ny = 10000
    rho = np.ones((ny, nx), dtype=dtype)
    u = np.ones((2, ny, nx), dtype=dtype)*1e-4
    nodetype = np.zeros((ny, nx), dtype=dtype)
    nodetype[0, :] = 1
    nodetype[-1, :] = 1
    f = np.random.rand(ny, nx, 9).astype(dtype)

    # GPU Memory Allocation
    f_d = cuda.to_device(f)
    rho_d = cuda.to_device(rho)
    u_d = cuda.to_device(u)
    nodetype_d = cuda.to_device(nodetype)

    #f[10,10,3] +=1e-3

    threads_per_block = (16, 16)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    compute_macro_vars_gpu[blocks_per_grid, threads_per_block](f_d, nodetype_d, rho_d, u_d)


    rho_gpu = rho_d.copy_to_host()
    u_gpu = u_d.copy_to_host()

    compute_macro_vars(f,nodetype,rho,u)

    #f_cpu[25] = 0

    equal = np.allclose(rho_gpu, rho, atol=1e-30)
    equal2 = np.allclose(u_gpu, u, atol=1e-30)
    print(equal,equal2)

test_macro()
