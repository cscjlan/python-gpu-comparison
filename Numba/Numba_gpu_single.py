import numpy as np
import numba as nb
from numba import cuda
import matplotlib.pyplot as plt
import time

dtype = np.float32
ex_host = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey_host = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
es_host = (1/3)**0.5
w0 = 4/9
ws = 1/9
wl = 1/36
w_host = np.array([w0, ws, ws, wl, wl, ws, ws, wl, wl], dtype=dtype)

ex = cuda.to_device(ex_host)
ey = cuda.to_device(ey_host)
#es = cuda.to_device(es_host)  # es is a scalar, no need to copy to constant memory
w = cuda.to_device(w_host)

@cuda.jit
def compute_macro_vars_gpu(f, nodetype, rho, u):
    j, i = cuda.grid(2)
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

@cuda.jit
def collide_gpu(f, rho, u, nodetype, tau, Fg):
    j, i = cuda.grid(2)
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
    j, i = cuda.grid(2)
    ny, nx = f.shape[0] , f.shape[1]  

    if i < ny and j < nx:
        for q in range(9):  # Copy all directions, including q=0
            f_old[i, j, q] = f[i, j, q]


# @cuda.jit
# def cuda_info(j):
#     j,i = cuda.grid(2)
#     print (j)

    


@cuda.jit
def stream_and_bounce_gpu(f, f_old, nodetype, ex, ey):
    j, i = cuda.grid(2)
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

              

def test_lb():
    nx = 1000
    ny = 1000
    niters = 400000
    rho = np.ones((ny, nx), dtype=dtype)
    tau = np.ones((ny, nx), dtype=dtype)
    u = np.zeros((2, ny, nx), dtype=dtype)
    Fg = np.zeros((2, ny, nx), dtype=dtype)
    Fg[0, :, :] = 1e-7
    nodetype = np.zeros((ny, nx), dtype=dtype)
    nodetype[0, :] = 1
    nodetype[-1, :] = 1
    f = np.zeros((ny, nx, 9), dtype=dtype)
    f_old = np.zeros((ny, nx, 9), dtype=dtype)

    x = 0.0

    # GPU Memory Allocation
    f_d = cuda.to_device(f)
    f_old_d = cuda.to_device(f_old)
    rho_d = cuda.to_device(rho)
    u_d = cuda.to_device(u)
    tau_d = cuda.to_device(tau)
    Fg_d = cuda.to_device(Fg)
    nodetype_d = cuda.to_device(nodetype)
    x_d = cuda.to_device(x)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)



    compute_edf_gpu[blocks_per_grid, threads_per_block](rho_d, u_d, nodetype_d, f_d,ex,ey,w)
    t0 = time.time()
    for i in range(niters):
        collide_gpu[blocks_per_grid, threads_per_block](f_d, rho_d, u_d, nodetype_d, tau_d, Fg_d)
        update_f_old[blocks_per_grid, threads_per_block](f_d, f_old_d)
        stream_and_bounce_gpu[blocks_per_grid, threads_per_block](f_d,f_old_d, nodetype_d,ex,ey)
        compute_macro_vars_gpu[blocks_per_grid, threads_per_block](f_d, nodetype_d, rho_d, u_d)
    t1 = time.time()

    f = f_d.copy_to_host()
    u = u_d.copy_to_host()

    mlups = (ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken", t1 - t0)

    # cuda_info[blocks_per_grid, threads_per_block](x_d)
    # xx = x_d.copy_to_host()
    # print(xx)

    # plt.figure(1)
    # plt.quiver(u[0], u[1])
    # plt.savefig("guiver.png", dpi=300)
    plt.figure(2)
    plt.plot(u[0][:, int(nx / 2)])
    plt.savefig("profile_gpu",dpi=300)

test_lb()
