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
nx, ny = 2000, 2000
niters = 100000
c_left = 1.0
c_right = 0.0
components = 5

# Lattice velocity directions
ex_host = np.array([0, 1, 0, -1, 0], dtype=dtype)
ey_host = np.array([0, 0, 1, 0, -1], dtype=dtype)
es_host = (1/3)**0.5
w_host = np.array([2/6, 1/6, 1/6, 1/6, 1/6], dtype=dtype)

# Taichi fields (GPU Memory)
ex = ti.field(dtype=ti.f32, shape=5)
ey = ti.field(dtype=ti.f32, shape=5)
w = ti.field(dtype=ti.f32, shape=5)

c = ti.field(dtype=ti.f32, shape=(components, ny, nx))
tau = ti.field(dtype=ti.f32, shape=(components, ny, nx))
u = ti.field(dtype=ti.f32, shape=(2, ny, nx))

Fg = ti.field(dtype=ti.f32, shape=(2, ny, nx))
nodetype = ti.field(dtype=ti.i32, shape=(ny, nx))
f = ti.field(dtype=ti.f32, shape=(components, 5, ny, nx))
#f_old = ti.field(dtype=ti.f32, shape=(ny, nx, 9))

# Copy constant data to GPU fields
ex.from_numpy(ex_host)
ey.from_numpy(ey_host)
w.from_numpy(w_host)



@ti.kernel
def compute_macro_vars_gpu_comp():
    for comp, i, j in ti.ndrange(components, ny, nx):  # Loop over components
        if nodetype[i, j] <= 0:
            c_ij = 0.0
            for q in ti.static(range(5)):
                c_ij += f[comp, q, i, j]
            c[comp, i, j] = c_ij
        else:
            c[comp, i, j] = 0

@ti.kernel
def compute_edf_gpu_comp():
    for comp, i, j in ti.ndrange(components, ny, nx):  # Loop over components
        if nodetype[i, j] <= 0:
            for q in ti.static(range(5)):
                es = (1 / 3) ** 0.5
                Termorder1 = (1.0 / es**2) * (ex[q] * u[0, i, j] + ey[q] * u[1, i, j])
                f[comp, q, i, j] = w[q] * c[comp, i, j] * (1 + Termorder1)


@ti.kernel
def collide_gpu_comp():
    for comp, i, j in ti.ndrange(components, ny, nx):  # Loop over components
        if nodetype[i, j] <= 0:

            # Compute equilibrium distribution function explicitly
            feq0 = c[comp, i, j] * (2.0 / 6.0)
            feq1 = c[comp, i, j] * (1.0 / 6.0) * (1.0 + 3.0 * u[0, i, j])
            feq2 = c[comp, i, j] * (1.0 / 6.0) * (1.0 + 3.0 * u[1, i, j])
            feq3 = c[comp, i, j] * (1.0 / 6.0) * (1.0 - 3.0 * u[0, i, j])
            feq4 = c[comp, i, j] * (1.0 / 6.0) * (1.0 - 3.0 * u[1, i, j])

            # Collision step
            f[comp, 0, i, j] = (1.0 - (1.0 / tau[comp, i, j])) * f[comp, 0, i, j] + (1.0 / tau[comp, i, j]) * feq0
            f[comp, 1, i, j] = (1.0 - (1.0 / tau[comp, i, j])) * f[comp, 1, i, j] + (1.0 / tau[comp, i, j]) * feq1
            f[comp, 2, i, j] = (1.0 - (1.0 / tau[comp, i, j])) * f[comp, 2, i, j] + (1.0 / tau[comp, i, j]) * feq2
            f[comp, 3, i, j] = (1.0 - (1.0 / tau[comp, i, j])) * f[comp, 3, i, j] + (1.0 / tau[comp, i, j]) * feq3
            f[comp, 4, i, j] = (1.0 - (1.0 / tau[comp, i, j])) * f[comp, 4, i, j] + (1.0 / tau[comp, i, j]) * feq4

            for q in ti.static(range(1, 3)):
                fswap = f[comp, q, i, j]
                f[comp, q, i, j] = f[comp, q + 2, i, j]
                f[comp, q + 2, i, j] = fswap

@ti.kernel
def stream_and_bounce_gpu_comp():
    for comp, i, j in ti.ndrange(components, ny, nx):  # Loop over components
        if nodetype[i, j] <= 0:
            for q in ti.static(range(1, 3)):
                nexti = int(i - ey[q])
                nextj = int(j + ex[q])
                if nexti > ny - 1: nexti = 0
                if nexti < 0: nexti = ny - 1
                if nextj > nx - 1: nextj = 0  
                if nextj < 0: nextj = nx - 1                      
                if nodetype[nexti, nextj] <= 0:
                    fswap = f[comp, q, nexti, nextj]
                    f[comp, q, nexti, nextj] = f[comp, q + 2, i, j]
                    f[comp, q + 2, i, j] = fswap

@ti.kernel
def boundary_conditions_comp():
    for comp, i, j in ti.ndrange(components, ny, nx):  # Loop over components
        if nodetype[i, 0] <= 0 and j == 0:
            f[comp, 1, i, 0] = c_left - f[comp, 0, i, 0] - f[comp, 2, i, 0] - f[comp, 3, i, 0] - f[comp, 4, i, 0]

        if nodetype[i, nx - 1] <= 0 and j == nx - 1:
            f[comp, 3, i, nx - 1] = c_right - f[comp, 0, i, nx - 1] - f[comp, 1, i, nx - 1] - f[comp, 2, i, nx - 1] - f[comp, 4, i, nx - 1]

def analytical_soln(cb,x,ts,tau, u):
    D = (tau - 0.5)/3.0
    c = (cb/2)*(erfc((x - u*ts)/(4*D*ts)**0.5)+
         erfc((x+ u * ts)/(4*D *ts)**0.5)*np.exp(u*x/D))
    return c


def test_lb():
    # Initialize Fields
    c_np = np.zeros((components, ny, nx), dtype=dtype)
    tau_np = np.ones((components, ny, nx), dtype=dtype)
    tau_np[0, :, :] = 1.0
    tau_np[1, :, :] = 0.9
    tau_np[2, :, :] = 0.8
    tau_np[3, :, :] = 0.7
    tau_np[4, :, :] = 0.6
    u_np = np.zeros((2, ny, nx), dtype=dtype)
    u_np[0, :, :] = 0.01
    nodetype_np = np.zeros((ny, nx), dtype=np.int32)
    f_np = np.zeros((components, 5, ny, nx), dtype=dtype)
    #f_old_np = np.zeros((ny, nx, 9), dtype=dtype)

    # Copy NumPy data to Taichi fields
    c.from_numpy(c_np)
    tau.from_numpy(tau_np)
    u.from_numpy(u_np)
    nodetype.from_numpy(nodetype_np)
    f.from_numpy(f_np)
    #f_old.from_numpy(f_old_np)

    compute_edf_gpu_comp()

    # Run Simulation
    t0 = time.time()
    for i in range(niters):
        collide_gpu_comp()
        #update_f_old()
        stream_and_bounce_gpu_comp()
        boundary_conditions_comp()
        compute_macro_vars_gpu_comp()
    t1 = time.time()

    # Copy results back
    c_np = c.to_numpy()
    c_analytical_01 = analytical_soln(c_left, np.arange(nx), niters, 1.0, 0.01)
    c_analytical_02 = analytical_soln(c_left, np.arange(nx), niters, 0.9, 0.01)
    c_analytical_03 = analytical_soln(c_left, np.arange(nx), niters, 0.8, 0.01)
    c_analytical_04 = analytical_soln(c_left, np.arange(nx), niters, 0.7, 0.01)
    c_analytical_05 = analytical_soln(c_left, np.arange(nx), niters, 0.6, 0.01)

    mlups = (components * ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken:", t1 - t0)

    plt.figure()
    plt.plot(c_np[0, int(ny / 2), :], label="LBM_01", color='blue', linestyle='-')
    plt.plot(c_np[1, int(ny / 2), :], label="LBM_02", color='green', linestyle='-')
    plt.plot(c_np[2, int(ny / 2), :], label="LBM_03", color='purple', linestyle='-')
    plt.plot(c_np[3, int(ny / 2), :], label="LBM_04", color='pink', linestyle='-')
    plt.plot(c_np[4, int(ny / 2), :], label="LBM_05", color='cyan', linestyle='-')
    plt.plot(c_analytical_01, label="Analytical Solution_01", color='red', linestyle='--')
    plt.plot(c_analytical_02, label="Analytical Solution_02", color='orange', linestyle='--')
    plt.plot(c_analytical_03, label="Analytical Solution_03", color='brown', linestyle='--')
    plt.plot(c_analytical_04, label="Analytical Solution_04", color='gray', linestyle='--')
    plt.plot(c_analytical_05, label="Analytical Solution_05", color='black', linestyle='--')
    plt.legend()
    plt.savefig("concentration_comp.png", dpi=300)

test_lb()
