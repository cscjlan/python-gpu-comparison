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
def collide(f,rho,u,nodetype,tau,Fg):
    ny,nx = nodetype.shape
    for i in nb.prange(ny):
        feq = [0.]*9
        for j in range(nx):
            if nodetype[i,j]<=0:
                #apply forcing
                u[0,i,j]+=Fg[0,i,j]*tau[i,j]/rho[i,j]
                u[1,i,j]+=Fg[1,i,j]*tau[i,j]/rho[i,j]
                #compute equilibrium distribution function
                feq[0]= rho[i, j]*(-2.0/3.0*u[0,i, j]**2 - 2.0/3.0*u[1,i, j]**2 + 4.0/9.0)
                feq[1]= rho[i, j]*((1.0/3.0)*u[0,i, j]**2 + (1.0/3.0)*u[0,i, j] - 1.0/6.0*u[1,i, j]**2 + 1.0/9.0)
                feq[2]= rho[i, j]*(-1.0/6.0*u[0,i, j]**2 + (1.0/3.0)*u[1,i, j]**2 + (1.0/3.0)*u[1,i, j] + 1.0/9.0)
                feq[3]= rho[i, j]*(-1.0/24.0*u[0,i, j]**2 + (1.0/12.0)*u[0,i, j] - 1.0/24.0*u[1,i, j]**2 + ( 
                1.0/12.0)*u[1,i, j] + (1.0/8.0)*(u[0,i, j] + u[1,i, j])**2 + 1.0/36.0)
                feq[4]= rho[i, j]*(-1.0/24.0*u[0,i, j]**2 + (1.0/12.0)*u[0,i, j] - 1.0/24.0*u[1,i,j]**2 - 
                1.0/12.0*u[1,i,j] + (1.0/8.0)*(u[0,i,j] - u[1,i,j])**2 + 1.0/36.0)
                feq[5]= rho[i,j]*((1.0/3.0)*u[0,i,j]**2 - 1.0/3.0*u[0,i,j] - 1.0/6.0*u[1,i,j]**2 + 1.0/9.0)
                feq[6]= rho[i,j]*(-1.0/6.0*u[0,i,j]**2 + (1.0/3.0)*u[1,i,j]**2 - 1.0/3.0*u[1,i,j] + 1.0/9.0)
                feq[7]= rho[i,j]*(-1.0/24.0*u[0,i,j]**2 - 1.0/12.0*u[0,i,j] - 1.0/24.0*u[1,i,j]**2 - 
                1.0/12.0*u[1,i,j] + (1.0/8.0)*(-u[0,i,j] - u[1,i,j])**2 + 1.0/36.0)
                feq[8]= rho[i,j]*(-1.0/24.0*u[0,i,j]**2 - 1.0/12.0*u[0,i,j] - 1.0/24.0*u[1,i,j]**2 + ( 
                1.0/12.0)*u[1,i,j] + (1.0/8.0)*(-u[0,i,j] + u[1,i,j])**2 + 1.0/36.0)
                #collision
                for q in range(9):
                    f[i,j,q]=(1.0-(1.0/(tau[i,j])))*f[i,j,q] +(1.0/(tau[i,j]))*feq[q] 

                for q in range(1,5):
                    fswap = f[i,j,q]
                    f[i,j,q]=f[i,j,q+4]
                    f[i,j,q+4]=fswap

    return f  


@jit_parallel
def stream_and_bounce(f,nodetype):
    global ex,ey,dtype
    ny,nx = nodetype.shape
    for i in nb.prange(ny):
        for j in range(nx):
            if nodetype[i,j]<=0:
                for q in range(1,5):
                    nexti = int(i-ey[q])
                    nextj = int(j+ex[q])
                    if nexti > ny-1: nexti = int(0)
                    if nextj > nx-1: nextj = int(0)                        
                    if nodetype[nexti,nextj]<=0:
                        fswap = f[nexti,nextj,q]
                        f[nexti,nextj,q] = f[i,j,q+4]
                        f[i,j,q+4] = fswap

    return f



@cuda.jit
def collide_gpu(f, rho, u, nodetype, tau, Fg):
    i, j = cuda.grid(2)
    ny, nx = nodetype.shape
    if i < ny and j < nx and nodetype[i, j] <= 0:
        # Apply forcing
        u[0, i, j] += Fg[0, i, j] * tau[i, j] / rho[i, j]
        u[1, i, j] += Fg[1, i, j] * tau[i, j] / rho[i, j]

        # Compute equilibrium distribution function explicitly
        feq0 = rho[i, j] * (-2.0/3.0 * u[0, i, j]**2 - 2.0/3.0 * u[1, i, j]**2 + 4.0/9.0)
        feq1 = rho[i, j] * ((1.0/3.0) * u[0, i, j]**2 + (1.0/3.0) * u[0, i, j] - 1.0/6.0 * u[1, i, j]**2 + 1.0/9.0)
        feq2 = rho[i, j] * (-1.0/6.0 * u[0, i, j]**2 + (1.0/3.0) * u[1, i, j]**2 + (1.0/3.0) * u[1, i, j] + 1.0/9.0)
        feq3 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 + (1.0/12.0) * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 + 
                            (1.0/12.0) * u[1, i, j] + (1.0/8.0) * (u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)
        feq4 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 + (1.0/12.0) * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 - 
                            1.0/12.0 * u[1, i, j] + (1.0/8.0) * (u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
        feq5 = rho[i, j] * ((1.0/3.0) * u[0, i, j]**2 - 1.0/3.0 * u[0, i, j] - 1.0/6.0 * u[1, i, j]**2 + 1.0/9.0)
        feq6 = rho[i, j] * (-1.0/6.0 * u[0, i, j]**2 + (1.0/3.0) * u[1, i, j]**2 - 1.0/3.0 * u[1, i, j] + 1.0/9.0)
        feq7 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/12.0 * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 - 
                            1.0/12.0 * u[1, i, j] + (1.0/8.0) * (-u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
        feq8 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/12.0 * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 + 
                            (1.0/12.0) * u[1, i, j] + (1.0/8.0) * (-u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)

        # Collision step
        f[i, j, 0] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 0] + (1.0 / tau[i, j]) * feq0
        f[i, j, 1] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 1] + (1.0 / tau[i, j]) * feq1
        f[i, j, 2] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 2] + (1.0 / tau[i, j]) * feq2
        f[i, j, 3] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 3] + (1.0 / tau[i, j]) * feq3
        f[i, j, 4] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 4] + (1.0 / tau[i, j]) * feq4
        f[i, j, 5] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 5] + (1.0 / tau[i, j]) * feq5
        f[i, j, 6] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 6] + (1.0 / tau[i, j]) * feq6
        f[i, j, 7] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 7] + (1.0 / tau[i, j]) * feq7
        f[i, j, 8] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 8] + (1.0 / tau[i, j]) * feq8


@cuda.jit
def update_f_old(f, f_old):
    i, j = cuda.grid(2)
    ny, nx = f.shape[0] , f.shape[1]  

    if i < ny and j < nx:
        for q in range(9):  # Copy all directions, including q=0
            f_old[i, j, q] = f[i, j, q]

@cuda.jit
def stream_and_bounce_gpu(f, f_old, nodetype, ex, ey):
    i, j = cuda.grid(2)
    ny, nx = nodetype.shape

    if i < ny and j < nx and nodetype[i, j] <= 0:
        for q in range(1, 9):  # Direction 0 does not need streaming
            per_i = (i + int(ey[q])) % ny  # Periodic boundary
            per_j = (j - int(ex[q])) % nx  # Periodic boundary
            
            if nodetype[per_i, per_j] <= 0:
                f[i, j, q] = f_old[per_i, per_j, q]  # Normal streaming
            else:  # Bounce-back condition
                if q < 5:
                    f[i, j, q] = f_old[i, j, q + 4]
                else:
                    f[i, j, q] = f_old[i, j, q - 4]


def test_stream():
    nx = 10000
    ny = 10000
    rho = np.ones((ny, nx), dtype=dtype)
    tau = np.ones((ny, nx), dtype=dtype)
    u = np.ones((2, ny, nx), dtype=dtype)*1e-4
    Fg = np.zeros((2, ny, nx), dtype=dtype)
    Fg[0, :, :] = 1e-4
    nodetype = np.zeros((ny, nx), dtype=dtype)
    nodetype[0, :] = 1
    nodetype[-1, :] = 1
    f = np.random.rand(ny, nx, 9).astype(dtype)
    f_old = np.zeros((ny, nx, 9), dtype=dtype)


    # GPU Memory Allocation
    f_d = cuda.to_device(f)
    f_old_d = cuda.to_device(f_old)
    rho_d = cuda.to_device(rho)
    u_d = cuda.to_device(u)
    tau_d = cuda.to_device(tau)
    Fg_d = cuda.to_device(Fg)
    nodetype_d = cuda.to_device(nodetype)

    #rho[10,10] += 1e-4

    threads_per_block = (16, 16)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    collide_gpu[blocks_per_grid, threads_per_block](f_d, rho_d, u_d, nodetype_d, tau_d, Fg_d)

    update_f_old[blocks_per_grid, threads_per_block](f_d,f_old_d)

    stream_and_bounce_gpu[blocks_per_grid, threads_per_block](f_d, f_old_d, nodetype_d, ex_gpu, ey_gpu)

    #f_d[10,10,2] +=1e-3

    f_gpu = f_d.copy_to_host()

    f_old_gpu = f_old_d.copy_to_host()

    collide(f, rho,u,nodetype, tau, Fg)

    f_cpu = stream_and_bounce(f, nodetype)

    #f_cpu[25] = 0

    # print("f_gpu dtype:", f_gpu.dtype)
    # print("f_cpu dtype:", f_cpu.dtype)

    # print("NaNs in GPU:", np.isnan(f_gpu).any())
    # print("NaNs in CPU:", np.isnan(f_cpu).any())



    equal = np.allclose(f_gpu, f_cpu, atol=1e-30)
    print(equal)

test_stream()