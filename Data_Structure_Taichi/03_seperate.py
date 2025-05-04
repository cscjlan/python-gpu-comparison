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

c1 = ti.field(dtype=ti.f32, shape=(ny, nx))
c2 = ti.field(dtype=ti.f32, shape=(ny, nx))
c3 = ti.field(dtype=ti.f32, shape=(ny, nx))
c4 = ti.field(dtype=ti.f32, shape=(ny, nx))
c5 = ti.field(dtype=ti.f32, shape=(ny, nx))
tau1 = ti.field(dtype=ti.f32, shape=(ny, nx))
tau2 = ti.field(dtype=ti.f32, shape=(ny, nx))
tau3 = ti.field(dtype=ti.f32, shape=(ny, nx))
tau4 = ti.field(dtype=ti.f32, shape=(ny, nx))
tau5 = ti.field(dtype=ti.f32, shape=(ny, nx))
u = ti.field(dtype=ti.f32, shape=(2, ny, nx))

Fg = ti.field(dtype=ti.f32, shape=(2, ny, nx))
nodetype = ti.field(dtype=ti.i32, shape=(ny, nx))
f1 = ti.field(dtype=ti.f32, shape=(5, ny, nx))
f2 = ti.field(dtype=ti.f32, shape=(5, ny, nx))
f3 = ti.field(dtype=ti.f32, shape=(5, ny, nx))
f4 = ti.field(dtype=ti.f32, shape=(5, ny, nx))
f5 = ti.field(dtype=ti.f32, shape=(5, ny, nx))
#f_old = ti.field(dtype=ti.f32, shape=(ny, nx, 9))

# Copy constant data to GPU fields
ex.from_numpy(ex_host)
ey.from_numpy(ey_host)
w.from_numpy(w_host)

@ti.kernel
def compute_macro_vars_gpu(f: ti.template(), c: ti.template()):
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
            c_ij = f[0, i, j] + f[1, i, j] + f[2, i, j] + f[3, i, j] + f[4, i, j]

            c[i, j] = c_ij
        else:
            c[i, j] = 0

@ti.kernel
def compute_edf_gpu(f: ti.template(), c: ti.template()):
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
            for q in range(5):
                es= (1/3)**0.5
                Termorder1 = (1. / es**2) * (ex[q] * u[0, i, j] + ey[q] * u[1, i, j])
                f[q, i, j] = w[q] * c[i, j] * (1 + Termorder1)

@ti.kernel
def collide_gpu(f: ti.template(), c: ti.template(), tau: ti.template()):
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
def stream_and_bounce_gpu(f: ti.template()):
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
def boundary_conditions(f: ti.template()):
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
    c1_np = np.zeros((ny, nx), dtype=dtype)
    c2_np = np.zeros((ny, nx), dtype=dtype)
    c3_np = np.zeros((ny, nx), dtype=dtype)
    c4_np = np.zeros((ny, nx), dtype=dtype)
    c5_np = np.zeros((ny, nx), dtype=dtype)
    tau1_np = np.ones((ny, nx), dtype=dtype)*1.0
    tau2_np = np.ones((ny, nx), dtype=dtype)*0.9
    tau3_np = np.ones((ny, nx), dtype=dtype)*0.8
    tau4_np = np.ones((ny, nx), dtype=dtype)*0.7
    tau5_np = np.ones((ny, nx), dtype=dtype)*0.6
    u_np = np.zeros((2, ny, nx), dtype=dtype)
    u_np[0, :, :] = 0.01
    nodetype_np = np.zeros((ny, nx), dtype=np.int32)
    f1_np = np.zeros((5, ny, nx), dtype=dtype)
    f2_np = np.zeros((5, ny, nx), dtype=dtype)
    f3_np = np.zeros((5, ny, nx), dtype=dtype)
    f4_np = np.zeros((5, ny, nx), dtype=dtype)
    f5_np = np.zeros((5, ny, nx), dtype=dtype)
    #f_old_np = np.zeros((ny, nx, 9), dtype=dtype)

    # Copy NumPy data to Taichi fields
    c1.from_numpy(c1_np)
    c2.from_numpy(c2_np)
    c3.from_numpy(c3_np)
    c4.from_numpy(c4_np)
    c5.from_numpy(c5_np)
    tau1.from_numpy(tau1_np)
    tau2.from_numpy(tau2_np)
    tau3.from_numpy(tau3_np)
    tau4.from_numpy(tau4_np)
    tau5.from_numpy(tau5_np)
    u.from_numpy(u_np)
    nodetype.from_numpy(nodetype_np)
    f1.from_numpy(f1_np)
    f2.from_numpy(f2_np)
    f3.from_numpy(f3_np)
    f4.from_numpy(f4_np)
    f5.from_numpy(f5_np)
    #f_old.from_numpy(f_old_np)

    compute_edf_gpu(f1, c1)
    compute_edf_gpu(f2, c2)
    compute_edf_gpu(f3, c3)
    compute_edf_gpu(f4, c4)
    compute_edf_gpu(f5, c5)

    # Run Simulation
    t0 = time.time()
    for i in range(niters):
        collide_gpu(f1, c1, tau1)
        collide_gpu(f2, c2, tau2)
        collide_gpu(f3, c3, tau3)
        collide_gpu(f4, c4, tau4)
        collide_gpu(f5, c5, tau5)
        #update_f_old()
        stream_and_bounce_gpu(f1)
        stream_and_bounce_gpu(f2)
        stream_and_bounce_gpu(f3)
        stream_and_bounce_gpu(f4)
        stream_and_bounce_gpu(f5)
        boundary_conditions(f1)
        boundary_conditions(f2)
        boundary_conditions(f3)
        boundary_conditions(f4)
        boundary_conditions(f5)
        compute_macro_vars_gpu(f1, c1)
        compute_macro_vars_gpu(f2, c2)
        compute_macro_vars_gpu(f3, c3)
        compute_macro_vars_gpu(f4, c4)
        compute_macro_vars_gpu(f5, c5)
    t1 = time.time()

    # Copy results back
    c1_np = c1.to_numpy()
    c2_np = c2.to_numpy()
    c3_np = c3.to_numpy()
    c4_np = c4.to_numpy()
    c5_np = c5.to_numpy()
    c1_analytical = analytical_soln(c_left, np.arange(nx), niters, (tau1_np[0,0] - 0.5)/3.0, 0.01)
    c2_analytical = analytical_soln(c_left, np.arange(nx), niters, (tau2_np[0,0] - 0.5)/3.0, 0.01)
    c3_analytical = analytical_soln(c_left, np.arange(nx), niters, (tau3_np[0,0] - 0.5)/3.0, 0.01)
    c4_analytical = analytical_soln(c_left, np.arange(nx), niters, (tau4_np[0,0] - 0.5)/3.0, 0.01)
    c5_analytical = analytical_soln(c_left, np.arange(nx), niters, (tau5_np[0,0] - 0.5)/3.0, 0.01)

    mlups = (components * ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken:", t1 - t0)

    plt.figure()
    plt.plot(c1_np[int(ny / 2), :], label="LBM01", color='blue', linestyle='-')
    plt.plot(c2_np[int(ny / 2), :], label="LBM02", color='green', linestyle='-')
    plt.plot(c3_np[int(ny / 2), :], label="LBM03", color='orange', linestyle='-')
    plt.plot(c4_np[int(ny / 2), :], label="LBM04", color='purple', linestyle='-')
    plt.plot(c5_np[int(ny / 2), :], label="LBM05", color='brown', linestyle='-')
    plt.plot(c1_analytical, label="Analytical01", color='blue', linestyle='--')
    plt.plot(c2_analytical, label="Analytical02", color='green', linestyle='--')
    plt.plot(c3_analytical, label="Analytical03", color='orange', linestyle='--')
    plt.plot(c4_analytical, label="Analytical04", color='purple', linestyle='--')
    plt.plot(c5_analytical, label="Analytical05", color='brown', linestyle='--')
    plt.title("Concentration Seperate")
    plt.legend()
    plt.savefig("concentration_seperate.png", dpi=300)

test_lb()
