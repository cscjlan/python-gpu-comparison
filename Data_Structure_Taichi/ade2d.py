import numpy as np
from scipy.special import erfc 
from math import exp
import taichi as ti
import time
import matplotlib.pyplot as plt

# Initialize Taichi on GPU
ti.init(arch=ti.gpu, device_memory_GB=4)  

# Constants
dtype = np.float32
nx, ny = 20000, 20000
niters = 200000
c_left = 1.0
c_right = 0.0

# Lattice velocity directions
ex_host = np.array([0, 1, 0, -1, 0], dtype=dtype)
ey_host = np.array([0, 0, 1, 0, -1], dtype=dtype)
es_host = (1/3)**0.5
w_host = np.array([2/6, 1/6, 1/6, 1/6, 1/6], dtype=dtype)

# Taichi fields (GPU Memory)
ex = ti.field(dtype=ti.f32, shape=5)
ey = ti.field(dtype=ti.f32, shape=5)
w = ti.field(dtype=ti.f32, shape=5)

c = ti.field(dtype=ti.f32, shape=(ny, nx))
tau = ti.field(dtype=ti.f32, shape=(ny, nx))
u = ti.field(dtype=ti.f32, shape=(2, ny, nx))

Fg = ti.field(dtype=ti.f32, shape=(2, ny, nx))
nodetype = ti.field(dtype=ti.i32, shape=(ny, nx))
f = ti.field(dtype=ti.f32, shape=(5, ny, nx))
#f_old = ti.field(dtype=ti.f32, shape=(ny, nx, 9))

# Copy constant data to GPU fields
ex.from_numpy(ex_host)
ey.from_numpy(ey_host)
w.from_numpy(w_host)

@ti.kernel
def compute_macro_vars_gpu():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
            c_ij = f[0, i, j] + f[1, i, j] + f[2, i, j] + f[3, i, j] + f[4, i, j]

            c[i, j] = c_ij
        else:
            c[i, j] = 0

@ti.kernel
def compute_edf_gpu():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
            for q in range(5):
                es= (1/3)**0.5
                Termorder1 = (1. / es**2) * (ex[q] * u[0, i, j] + ey[q] * u[1, i, j])
                f[q, i, j] = w[q] * c[i, j] * (1 + Termorder1)

@ti.kernel
def collide_gpu():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:

            # Compute equilibrium distribution function explicitly
            feq0 = c[i, j] * (2.0/6.0)
            feq1 = c[i, j] * (1.0/6.0)*(1.0 + 3.0 * u[0,i,j])
            feq2 = c[i, j] * (1.0/6.0)*(1.0 + 3.0 * u[1,i,j])
            feq3 = c[i, j] * (1.0/6.0)*(1.0 - 3.0 * u[0,i,j])
            feq4 = c[i, j] * (1.0/6.0)*(1.0 - 3.0 * u[1,i,j])

            # Collision step
            f[0, i, j] = (1.0 - (1.0 / tau[i, j])) * f[0, i, j] + (1.0 / tau[i, j]) * feq0
            f[1, i, j] = (1.0 - (1.0 / tau[i, j])) * f[1, i, j] + (1.0 / tau[i, j]) * feq1
            f[2, i, j] = (1.0 - (1.0 / tau[i, j])) * f[2, i, j] + (1.0 / tau[i, j]) * feq2
            f[3, i, j] = (1.0 - (1.0 / tau[i, j])) * f[3, i, j] + (1.0 / tau[i, j]) * feq3
            f[4, i, j] = (1.0 - (1.0 / tau[i, j])) * f[4, i, j] + (1.0 / tau[i, j]) * feq4

            for q in ti.static(range(1,3)):
                fswap = f[q, i,j]
                f[q, i,j]=f[q+2,i,j]
                f[q+2,i,j]=fswap


@ti.kernel
def stream_and_bounce_gpu():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
            for q in ti.static(range(1,3)):
                nexti = int(i-ey[q])
                nextj = int(j+ex[q])
                if nexti > ny-1: nexti = int(0)
                if nexti < 0: nexti = int(ny-1)
                if nextj > nx-1: nextj = int(0)  
                if nextj < 0: nextj = int(nx-1)                      
                if nodetype[nexti,nextj]<=0:
                    fswap = f[q,nexti,nextj]
                    f[q,nexti,nextj] = f[q+2,i,j]
                    f[q+2,i,j] = fswap

@ti.kernel
def boundary_conditions():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, 0] <=0:
            f[1, i, 0] = c_left - f[0, i, 0] - f[2, i, 0] - f[3, i, 0] - f[4, i, 0]

        if nodetype[i, nx-1] <=0:
            f[3, i, nx-1] = c_right - f[0, i, nx-1] - f[1, i, nx-1] - f[2, i, nx-1] - f[4, i, nx-1]

def analytical_soln(cb,x,ts,D, u):
    c = (cb/2)*(erfc((x - u*ts)/(4*D*ts)**0.5)+
         erfc((x+ u * ts)/(4*D *ts)**0.5)*np.exp(u*x/D))
    return c

def test_lb():
    # Initialize Fields
    c_np = np.zeros((ny, nx), dtype=dtype)
    tau_np = np.ones((ny, nx), dtype=dtype)
    u_np = np.zeros((2, ny, nx), dtype=dtype)
    u_np[0, :, :] = 0.01
    nodetype_np = np.zeros((ny, nx), dtype=np.int32)
    f_np = np.zeros((5, ny, nx), dtype=dtype)
    #f_old_np = np.zeros((ny, nx, 9), dtype=dtype)

    # Copy NumPy data to Taichi fields
    c.from_numpy(c_np)
    tau.from_numpy(tau_np)
    u.from_numpy(u_np)
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
        boundary_conditions()
        compute_macro_vars_gpu()
    t1 = time.time()

    # Copy results back
    c_np = c.to_numpy()
    c_analytical = analytical_soln(c_left, np.arange(nx), niters, 1.0/6.0, 0.01)

    mlups = (ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken:", t1 - t0)

    plt.figure()
    plt.plot(c_np[int(ny / 2), :], label="LBM", color='blue', linestyle='-')
    plt.plot(c_analytical, label="Analytical Solution", color='red', linestyle='--')
    plt.legend()
    plt.savefig("concentration.png", dpi=300)

test_lb()
