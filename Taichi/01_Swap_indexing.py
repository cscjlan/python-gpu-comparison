import numpy as np
import taichi as ti
import time
import matplotlib.pyplot as plt

# Initialize Taichi on GPU
ti.init(arch=ti.gpu)  

# Constants
dtype = np.float32
nx, ny = 2000, 1000
niters = 400

# Lattice velocity directions
ex_host = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey_host = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
es_host = (1/3)**0.5
w_host = np.array([4/9, 1/9, 1/9, 1/36, 1/36, 1/9, 1/9, 1/36, 1/36], dtype=dtype)

# Taichi fields (GPU Memory)
ex = ti.field(dtype=ti.f32, shape=9)
ey = ti.field(dtype=ti.f32, shape=9)
w = ti.field(dtype=ti.f32, shape=9)

rho = ti.field(dtype=ti.f32, shape=(ny, nx))
tau = ti.field(dtype=ti.f32, shape=(ny, nx))
u = ti.field(dtype=ti.f32, shape=(2, ny, nx))
Fg = ti.field(dtype=ti.f32, shape=(2, ny, nx))
nodetype = ti.field(dtype=ti.i32, shape=(ny, nx))
f = ti.field(dtype=ti.f32, shape=(9, ny, nx))
#f_old = ti.field(dtype=ti.f32, shape=(ny, nx, 9))

# Copy constant data to GPU fields
ex.from_numpy(ex_host)
ey.from_numpy(ey_host)
w.from_numpy(w_host)

@ti.kernel
def compute_macro_vars_gpu():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
            rho_ij = f[0, i, j] + f[1, i, j] + f[2, i, j] + f[3, i, j] + f[4, i, j] + f[5, i, j] + f[6, i, j] + f[7, i, j] + f[8, i, j]
            fdotex = f[1, i, j] + f[3, i, j] + f[4, i, j] - f[5, i, j] - f[7, i, j] - f[8, i, j]
            fdotey = f[2, i, j] + f[3, i, j] - f[4, i, j] - f[6, i, j] - f[7, i, j] + f[8, i, j]

            rho[i, j] = rho_ij
            u[0, i, j] = fdotex / rho_ij
            u[1, i, j] = fdotey / rho_ij
        else:
            rho[i, j] = 0
            u[0, i, j] = 0
            u[1, i, j] = 0

@ti.kernel
def compute_edf_gpu():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
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
                f[q, i, j] = w[q] * rho[i, j] * (1 + Termorder1 + Termorder2)

@ti.kernel
def collide_gpu():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
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
            f[0, i, j] = (1.0 - (1.0 / tau[i, j])) * f[0, i, j] + (1.0 / tau[i, j]) * feq0
            f[1, i, j] = (1.0 - (1.0 / tau[i, j])) * f[1, i, j] + (1.0 / tau[i, j]) * feq1
            f[2, i, j] = (1.0 - (1.0 / tau[i, j])) * f[2, i, j] + (1.0 / tau[i, j]) * feq2
            f[3, i, j] = (1.0 - (1.0 / tau[i, j])) * f[3, i, j] + (1.0 / tau[i, j]) * feq3
            f[4, i, j] = (1.0 - (1.0 / tau[i, j])) * f[4, i, j] + (1.0 / tau[i, j]) * feq4
            f[5, i, j] = (1.0 - (1.0 / tau[i, j])) * f[5, i, j] + (1.0 / tau[i, j]) * feq5
            f[6, i, j] = (1.0 - (1.0 / tau[i, j])) * f[6, i, j] + (1.0 / tau[i, j]) * feq6
            f[7, i, j] = (1.0 - (1.0 / tau[i, j])) * f[7, i, j] + (1.0 / tau[i, j]) * feq7
            f[8, i, j] = (1.0 - (1.0 / tau[i, j])) * f[8, i, j] + (1.0 / tau[i, j]) * feq8

            for q in ti.static(range(1,5)):
                fswap = f[q, i,j]
                f[q, i,j]=f[q+4,i,j]
                f[q+4,i,j]=fswap


# @ti.kernel
# def update_f_old():
#     for i, j, q in ti.ndrange(ny, nx, 9):
#         f_old[i, j, q] = f[i, j, q]

@ti.kernel
def stream_and_bounce_gpu():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
            for q in ti.static(range(1,5)):
                nexti = int(i-ey[q])
                nextj = int(j+ex[q])
                if nexti > ny-1: nexti = int(0)
                if nextj > nx-1: nextj = int(0)                        
                if nodetype[nexti,nextj]<=0:
                    fswap = f[q,nexti,nextj]
                    f[q,nexti,nextj] = f[q+4,i,j]
                    f[q+4,i,j] = fswap

def test_lb():
    # Initialize Fields
    rho_np = np.ones((ny, nx), dtype=dtype)
    tau_np = np.ones((ny, nx), dtype=dtype)
    u_np = np.zeros((2, ny, nx), dtype=dtype)
    Fg_np = np.zeros((2, ny, nx), dtype=dtype)
    Fg_np[0, :, :] = 1e-7
    nodetype_np = np.zeros((ny, nx), dtype=np.int32)
    nodetype_np[0, :] = 1
    nodetype_np[-1, :] = 1
    f_np = np.zeros((9, ny, nx), dtype=dtype)
    #f_old_np = np.zeros((ny, nx, 9), dtype=dtype)

    # Copy NumPy data to Taichi fields
    rho.from_numpy(rho_np)
    tau.from_numpy(tau_np)
    u.from_numpy(u_np)
    Fg.from_numpy(Fg_np)
    nodetype.from_numpy(nodetype_np)
    f.from_numpy(f_np)
    #f_old.from_numpy(f_old_np)

    compute_edf_gpu()

    # Run Simulation
    t0 = time.time()
    for i in range(niters):
        collide_gpu()
        #update_f_old()
        stream_and_bounce_gpu()
        compute_macro_vars_gpu()
    t1 = time.time()

    # Copy results back
    u_np = u.to_numpy()
    f_np = f.to_numpy()

    mlups = (ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken:", t1 - t0)

    plt.figure()
    plt.plot(u_np[0][:, int(nx / 2)])
    plt.savefig("profile_taichi_swap.png", dpi=300)

test_lb()
