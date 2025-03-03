import numpy as np
import numba as nb
from numba import cuda
import matplotlib.pyplot as plt
import time
dtype = np.float64
jit_parallel = nb.njit(nogil=True,parallel=True)


dtype = np.float64
ex = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
es = (1/3)**0.5
w0 = 4/9
ws = 1/9
wl = 1/36
w = np.array([w0, ws, ws, wl, wl, ws, ws, wl, wl], dtype=dtype)

ex_gpu = cuda.to_device(ex)
ey_gpu = cuda.to_device(ey)
#es = cuda.to_device(es_host)  # es is a scalar, no need to copy to constant memory
w_gpu = cuda.to_device(w)


@jit_parallel
def compute_edf(rho,u,nodetype):
    global ex,ey,w,dtype
    ny,nx = nodetype.shape
    feq = np.zeros((ny,nx,9),dtype=dtype)
    for i in nb.prange(ny):
        for j in range(nx):
            if nodetype[i,j]<=0:
                for q in range(9):
                    Termorder1  = (1./es**2)*(ex[q]*u[0,i,j] + ey[q]*u[1,i,j])
                    euxy        = ex[q]*ey[q]*u[0,i,j]*u[1,i,j]
                    euxx        = ex[q]*ex[q]*u[0,i,j]*u[0,i,j]
                    euyy        = ey[q]*ey[q]*u[1,i,j]*u[1,i,j]
                    eu2         = 2*euxy + euxx + euyy
                    ux2         = u[0,i,j]*u[0,i,j]
                    uy2         = u[1,i,j]*u[1,i,j]
                    u2          = ux2+uy2
                    Termorder2  =(0.5/es**4)*eu2 - (0.5/es**2)*u2     
                    feq[i,j,q]= w[q]*rho[i,j]*(1+ Termorder1+ Termorder2)     
    return feq



@cuda.jit
def compute_edf_gpu(rho, u, nodetype, feq,ex,ey,w):
    j, i = cuda.grid(2)
    ny, nx = nodetype.shape
    if i < ny and j < nx and nodetype[i, j] <= 0:
        for q in range(9):
            es= (1/3)**0.5
            Termorder1 = (1. / es**2) * (ex[q] * u[0, i, j] + ey[q] * u[1, i, j])
            euxy = ex[q] * ey[q] * u[0, i, j] * u[1, i, j]
            euxx = ex[q] * ex[q] * u[0, i, j] * u[0, i, j]
            euyy = ey[q] * ey[q] * u[1, i, j] * u[1, i, j]
            eu2 = 2 * euxy + euxx + euyy
            ux2 = u[0, i, j] * u[0, i, j]
            uy2 = u[1, i, j] * u[1, i, j]
            u2 = ux2 + uy2
            Termorder2 = (0.5 / es**4) * eu2 - (0.5 / es**2) * u2
            feq[i, j, q] = w[q] * rho[i, j] * (1 + Termorder1 + Termorder2)



def test_stream():
    nx = 10000
    ny = 20000
    rho = np.ones((ny, nx), dtype=dtype)
    u = np.ones((2, ny, nx), dtype=dtype)*1e-4
    nodetype = np.zeros((ny, nx), dtype=dtype)
    nodetype[0, :] = 1
    nodetype[-1, :] = 1
    f = np.zeros((ny, nx, 9), dtype=dtype)

    # GPU Memory Allocation
    f_d = cuda.to_device(f)
    rho_d = cuda.to_device(rho)
    u_d = cuda.to_device(u)
    nodetype_d = cuda.to_device(nodetype)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    compute_edf_gpu[blocks_per_grid, threads_per_block](rho_d, u_d, nodetype_d, f_d,ex_gpu,ey_gpu,w_gpu)


    f_gpu = f_d.copy_to_host()

    f_cpu = compute_edf(rho,u,nodetype)

    #f_cpu[25] = 0

    equal = np.allclose(f_gpu, f_cpu, atol=1e-17, rtol=0)
    print(equal)

test_stream()



