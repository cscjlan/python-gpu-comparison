import numpy as np
import taichi as ti
import time
import matplotlib.pyplot as plt

# Initialize Taichi on GPU
ti.init(arch=ti.gpu)  

# Constants
dtype = np.float32
nz, ny, nx = 1000, 300, 300
niters = 100000

# Lattice velocity directions
ex_host = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0, -1, 0, 0, -1, -1, -1, -1, 0, 0], dtype=dtype)
ey_host = np.array([0, 0, 1, 0, 0, 0, 1, -1, 1, -1, 0, -1, 0, 0, 0, -1, 1, -1, 1], dtype=dtype)
ez_host = np.array([0, 0, 0, 1, 1, -1, 0, 0, 1, 1, 0, 0, -1, -1, 1, 0, 0, -1, -1], dtype=dtype)
es_host = (1/3)**0.5
w_host = np.array([1/3, 1/18, 1/18, 1/18, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/18, 1/18, 1/18, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36], dtype=dtype)

# Taichi fields (GPU Memory)
ex = ti.field(dtype=ti.f32, shape=19)
ey = ti.field(dtype=ti.f32, shape=19)
ez = ti.field(dtype=ti.f32, shape=19)
w = ti.field(dtype=ti.f32, shape=19)

rho = ti.field(dtype=ti.f32, shape=(nz, ny, nx))
tau = ti.field(dtype=ti.f32, shape=(nz, ny, nx))
u = ti.field(dtype=ti.f32, shape=(3, nz, ny, nx))
Fg = ti.field(dtype=ti.f32, shape=(3, nz, ny, nx))
nodetype = ti.field(dtype=ti.i32, shape=(nz, ny, nx))
f = ti.field(dtype=ti.f32, shape=(19, nz, ny, nx))
#f_old = ti.field(dtype=ti.f32, shape=(ny, nx, 9))

# Copy constant data to GPU fields
ex.from_numpy(ex_host)
ey.from_numpy(ey_host)
ez.from_numpy(ez_host)
w.from_numpy(w_host)

@ti.kernel
def compute_macro_vars_gpu():
    for I in ti.grouped(rho):
        i, j, k = I
        if nodetype[I] <= 0:
            rho_ijk = f[0, I] + f[1, I] + f[2, I] + f[3, I] + f[4, I] + f[5, I] + f[6, I] + f[7, I] + f[8, I] + f[9, I] + f[10, I] + f[11, I] + f[12, I] + f[13, I] + f[14, I] + f[15, I] + f[16, I] + f[17, I] + f[18, I]
            fdotex = f[1, I] + f[4, I] + f[5, I] + f[6, I] + f[7, I] - f[10, I] - f[13, I] - f[14, I]- f[15, I] - f[16, I]
            fdotey = f[2, I] + f[6, I] - f[7, I] + f[8, I] - f[9, I] - f[11, I] - f[15, I] + f[16, I] - f[17, I] + f[18, I]
            fdotez = f[3, I] + f[4, I] - f[5, I] + f[8, I] + f[9, I] - f[12, I] - f[13, I] + f[14, I] - f[17, I] - f[18, I]

            rho[I] = rho_ijk
            u[0, I] = fdotex / rho_ijk
            u[1, I] = fdotey / rho_ijk
            u[2, I] = fdotez / rho_ijk
        else:
            rho[I] = 0
            u[0, I] = 0
            u[1, I] = 0
            u[2, I] = 0

@ti.kernel
def compute_edf_gpu():
    for I in ti.grouped(rho):
        i, j, k = I
        if nodetype[I] <= 0:
            f[0, I]= rho[I]*(-1.0/2.0*u[0,I]**2 - 1.0/2.0*u[1,I]**2 - 1.0/2.0*u[2,I]**2 + 1.0/3.0)
            f[1, I]= rho[I]*((1.0/6.0)*u[0, I]**2 + (1.0/6.0)*u[0, I] - 1.0/12.0*u[1, I]**2 - 1.0/12.0*u[2, I]**2 + 1.0/18.0)
            f[2, I]= rho[I]*(-1.0/12.0*u[0, I]**2 + (1.0/6.0)*u[1, I]**2 + (1.0/6.0)*u[1, I] - 1.0/12.0*u[2, I]**2 + 1.0/18.0)
            f[3, I]= rho[I]*(-1.0/12.0*u[0, I]**2 - 1.0/12.0*u[1, I]**2 + (1.0/6.0)*u[2, I]**2 + (1.0/6.0)*u[2, I] + 1.0/18.0)
            f[4, I]= rho[I]*(-1.0/24.0*u[0, I]**2 + (1.0/12.0)*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/24.0*u[2, I]**2 + (1.0/12.0)*u[2, I] + (1.0/8.0)*(u[0, I] + u[2, I])**2 + 1.0/36.0)
            f[5, I]= rho[I]*(-1.0/24.0*u[0, I]**2 + (1.0/12.0)*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/24.0*u[2, I]**2 - 1.0/12.0*u[2, I] + (1.0/8.0)*(u[0, I] - u[2, I] )**2 + 1.0/36.0)
            f[6, I]= rho[I]*(-1.0/24.0*u[0, I]**2 + (1.0/12.0)*u[0, I] - 1.0/24.0*u[1, I]**2 + (1.0/12.0)*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/8.0)*(u[0, I] + u[1, I])**2 + 1.0/36.0)
            f[7, I]= rho[I]*(-1.0/24.0*u[0, I]**2 + (1.0/12.0)*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/12.0*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/8.0)*(u[0, I] - u[1, I])**2 + 1.0/36.0)
            f[8, I]= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/24.0*u[1, I]**2 + (1.0/12.0)*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/12.0)*u[2, I] + (1.0/8.0)*(u[1, I] + u[2, I])**2 + 1.0/36.0)
            f[9, I]= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/24.0*u[1, I]**2 - 1.0/12.0*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/12.0)*u[2, I] + (1.0/8.0)*(-u[1, I] + u[2, I])**2 + 1.0/36.0)
            f[10, I]= rho[I]*((1.0/6.0)*u[0, I]**2 - 1.0/6.0*u[0, I] - 1.0/12.0*u[1, I]**2 - 1.0/12.0*u[2, I]**2 + 1.0/18.0)
            f[11, I]= rho[I]*(-1.0/12.0*u[0, I]**2 + (1.0/6.0)*u[1, I]**2 - 1.0/6.0*u[1, I] - 1.0/12.0*u[2, I]**2 + 1.0/18.0)
            f[12, I]= rho[I]*(-1.0/12.0*u[0, I]**2 - 1.0/12.0*u[1, I]**2 + (1.0/6.0)*u[2, I]**2 - 1.0/6.0*u[2, I] + 1.0/18.0)
            f[13, I]= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/12.0*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/24.0*u[2, I]**2 - 1.0/12.0*u[2, I] + (1.0/8.0)*(-u[0, I] - u[2, I])**2 + 1.0/36.0)
            f[14, I]= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/12.0*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/24.0*u[2, I]**2 + (1.0/12.0)*u[2, I] + (1.0/8.0)*(-u[0, I] + u[2, I])**2 + 1.0/36.0)
            f[15, I]= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/12.0*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/12.0*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/8.0)*(-u[0, I] - u[1, I])**2 + 1.0/36.0)
            f[16, I]= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/12.0*u[0, I] - 1.0/24.0*u[1, I]**2 + (1.0/12.0)*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/8.0)*(-u[0, I] + u[1, I])**2 + 1.0/36.0)
            f[17, I]= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/24.0*u[1, I]**2 - 1.0/12.0*u[1, I] - 1.0/24.0*u[2, I]**2 - 1.0/12.0*u[2, I] + (1.0/8.0)*(-u[1, I] - u[2, I])**2 + 1.0/36.0)
            f[18, I]= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/24.0*u[1, I]**2 + (1.0/12.0)*u[1, I] - 1.0/24.0*u[2, I]**2 - 1.0/12.0*u[2, I] + (1.0/8.0)*(u[1, I] - u[2, I])**2 + 1.0/36.0)

@ti.kernel
def collide_gpu():
    for I in ti.grouped(rho):
        i, j, k = I
        if nodetype[I] <= 0:
            u[0, I] += Fg[0, I] * tau[I] / rho[I]
            u[1, I] += Fg[1, I] * tau[I] / rho[I]
            u[2, I] += Fg[2, I] * tau[I] / rho[I]

            # Compute equilibrium distribution function explicitly
            feq0= rho[I]*(-1.0/2.0*u[0,I]**2 - 1.0/2.0*u[1,I]**2 - 1.0/2.0*u[2,I]**2 + 1.0/3.0)
            feq1= rho[I]*((1.0/6.0)*u[0, I]**2 + (1.0/6.0)*u[0, I] - 1.0/12.0*u[1, I]**2 - 1.0/12.0*u[2, I]**2 + 1.0/18.0)
            feq2= rho[I]*(-1.0/12.0*u[0, I]**2 + (1.0/6.0)*u[1, I]**2 + (1.0/6.0)*u[1, I] - 1.0/12.0*u[2, I]**2 + 1.0/18.0)
            feq3 = rho[I]*(-1.0/12.0*u[0, I]**2 - 1.0/12.0*u[1, I]**2 + (1.0/6.0)*u[2, I]**2 + (1.0/6.0)*u[2, I] + 1.0/18.0)
            feq4= rho[I]*(-1.0/24.0*u[0, I]**2 + (1.0/12.0)*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/24.0*u[2, I]**2 + (1.0/12.0)*u[2, I] + (1.0/8.0)*(u[0, I] + u[2, I])**2 + 1.0/36.0)
            feq5= rho[I]*(-1.0/24.0*u[0, I]**2 + (1.0/12.0)*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/24.0*u[2, I]**2 - 1.0/12.0*u[2, I] + (1.0/8.0)*(u[0, I] - u[2, I] )**2 + 1.0/36.0)
            feq6= rho[I]*(-1.0/24.0*u[0, I]**2 + (1.0/12.0)*u[0, I] - 1.0/24.0*u[1, I]**2 + (1.0/12.0)*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/8.0)*(u[0, I] + u[1, I])**2 + 1.0/36.0)
            feq7= rho[I]*(-1.0/24.0*u[0, I]**2 + (1.0/12.0)*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/12.0*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/8.0)*(u[0, I] - u[1, I])**2 + 1.0/36.0)
            feq8= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/24.0*u[1, I]**2 + (1.0/12.0)*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/12.0)*u[2, I] + (1.0/8.0)*(u[1, I] + u[2, I])**2 + 1.0/36.0)
            feq9= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/24.0*u[1, I]**2 - 1.0/12.0*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/12.0)*u[2, I] + (1.0/8.0)*(-u[1, I] + u[2, I])**2 + 1.0/36.0)
            feq10= rho[I]*((1.0/6.0)*u[0, I]**2 - 1.0/6.0*u[0, I] - 1.0/12.0*u[1, I]**2 - 1.0/12.0*u[2, I]**2 + 1.0/18.0)
            feq11= rho[I]*(-1.0/12.0*u[0, I]**2 + (1.0/6.0)*u[1, I]**2 - 1.0/6.0*u[1, I] - 1.0/12.0*u[2, I]**2 + 1.0/18.0)
            feq12= rho[I]*(-1.0/12.0*u[0, I]**2 - 1.0/12.0*u[1, I]**2 + (1.0/6.0)*u[2, I]**2 - 1.0/6.0*u[2, I] + 1.0/18.0)
            feq13= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/12.0*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/24.0*u[2, I]**2 - 1.0/12.0*u[2, I] + (1.0/8.0)*(-u[0, I] - u[2, I])**2 + 1.0/36.0)
            feq14= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/12.0*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/24.0*u[2, I]**2 + (1.0/12.0)*u[2, I] + (1.0/8.0)*(-u[0, I] + u[2, I])**2 + 1.0/36.0)
            feq15= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/12.0*u[0, I] - 1.0/24.0*u[1, I]**2 - 1.0/12.0*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/8.0)*(-u[0, I] - u[1, I])**2 + 1.0/36.0)
            feq16= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/12.0*u[0, I] - 1.0/24.0*u[1, I]**2 + (1.0/12.0)*u[1, I] - 1.0/24.0*u[2, I]**2 + (1.0/8.0)*(-u[0, I] + u[1, I])**2 + 1.0/36.0)
            feq17= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/24.0*u[1, I]**2 - 1.0/12.0*u[1, I] - 1.0/24.0*u[2, I]**2 - 1.0/12.0*u[2, I] + (1.0/8.0)*(-u[1, I] - u[2, I])**2 + 1.0/36.0)
            feq18= rho[I]*(-1.0/24.0*u[0, I]**2 - 1.0/24.0*u[1, I]**2 + (1.0/12.0)*u[1, I] - 1.0/24.0*u[2, I]**2 - 1.0/12.0*u[2, I] + (1.0/8.0)*(u[1, I] - u[2, I])**2 + 1.0/36.0)


            # Collision step
            f[0, I] = (1.0 - (1.0 / tau[I])) * f[0, I] + (1.0 / tau[I]) * feq0
            f[1, I] = (1.0 - (1.0 / tau[I])) * f[1, I] + (1.0 / tau[I]) * feq1
            f[2, I] = (1.0 - (1.0 / tau[I])) * f[2, I] + (1.0 / tau[I]) * feq2
            f[3, I] = (1.0 - (1.0 / tau[I])) * f[3, I] + (1.0 / tau[I]) * feq3
            f[4, I] = (1.0 - (1.0 / tau[I])) * f[4, I] + (1.0 / tau[I]) * feq4
            f[5, I] = (1.0 - (1.0 / tau[I])) * f[5, I] + (1.0 / tau[I]) * feq5
            f[6, I] = (1.0 - (1.0 / tau[I])) * f[6, I] + (1.0 / tau[I]) * feq6
            f[7, I] = (1.0 - (1.0 / tau[I])) * f[7, I] + (1.0 / tau[I]) * feq7
            f[8, I] = (1.0 - (1.0 / tau[I])) * f[8, I] + (1.0 / tau[I]) * feq8
            f[9, I] = (1.0 - (1.0 / tau[I])) * f[9, I] + (1.0 / tau[I]) * feq9
            f[10, I] = (1.0 - (1.0 / tau[I])) * f[10, I] + (1.0 / tau[I]) * feq10
            f[11, I] = (1.0 - (1.0 / tau[I])) * f[11, I] + (1.0 / tau[I]) * feq11
            f[12, I] = (1.0 - (1.0 / tau[I])) * f[12, I] + (1.0 / tau[I]) * feq12
            f[13, I] = (1.0 - (1.0 / tau[I])) * f[13, I] + (1.0 / tau[I]) * feq13
            f[14, I] = (1.0 - (1.0 / tau[I])) * f[14, I] + (1.0 / tau[I]) * feq14
            f[15, I] = (1.0 - (1.0 / tau[I])) * f[15, I] + (1.0 / tau[I]) * feq15
            f[16, I] = (1.0 - (1.0 / tau[I])) * f[16, I] + (1.0 / tau[I]) * feq16
            f[17, I] = (1.0 - (1.0 / tau[I])) * f[17, I] + (1.0 / tau[I]) * feq17
            f[18, I] = (1.0 - (1.0 / tau[I])) * f[18, I] + (1.0 / tau[I]) * feq18

            for q in ti.static(range(1,10)):
                fswap = f[q, I]
                f[q, I]=f[q+9,I]
                f[q+9,I]=fswap


# @ti.kernel
# def update_f_old():
#     for i, j, q in ti.ndrange(ny, nx, 9):
#         f_old[i, j, q] = f[i, j, q]

@ti.kernel
def stream_and_bounce_gpu():
    for I in ti.grouped(rho):
        i, j, k = I
        if nodetype[I] <= 0:
            for q in ti.static(range(1, 10)):
                nexti = ti.cast(i - ez[q], ti.i32)  # Explicit integer casting
                nextj = ti.cast(j - ey[q], ti.i32)
                nextk = ti.cast(k + ex[q], ti.i32)

                # Apply periodic boundary conditions efficiently
                nexti = ti.select(nexti >= nz, 0, nexti)
                nextj = ti.select(nextj >= ny, 0, nextj)
                nextk = ti.select(nextk >= nx, 0, nextk)

                if nodetype[nexti, nextj, nextk] <= 0:
                    # Perform streaming swap
                    f[q, nexti, nextj, nextk], f[q + 9, I] = f[q + 9, I], f[q, nexti, nextj, nextk]

def test_lb():
    # Initialize Fields
    rho_np = np.ones((nz, ny, nx), dtype=dtype)
    tau_np = np.ones((nz, ny, nx), dtype=dtype)
    u_np = np.zeros((3, nz, ny, nx), dtype=dtype)
    Fg_np = np.zeros((3, nz, ny, nx), dtype=dtype)
    Fg_np[0, :, :, :] = 1e-7
    nodetype_np = np.zeros((nz, ny, nx), dtype=np.int32)
    nodetype_np[0, :, :] = 1
    nodetype_np[-1, :, :] = 1
    f_np = np.zeros((19, nz, ny, nx), dtype=dtype)
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

    mlups = (nz * ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken:", t1 - t0)

    #Fg_np[0][10:30, :, :] = 0

    plt.figure()
    plt.plot(u_np[0][:, int(ny/2), int(nx/2)])
    plt.savefig("profile_taichi_swap_unroll.png", dpi=300)
    
    # plt.figure()
    # plt.imshow(u_np[0][:,:,:])
    # plt.savefig("field_taichi_swap_unroll.png", dpi=300)

test_lb()
