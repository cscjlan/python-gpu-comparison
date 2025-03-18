import numpy as np
import taichi as ti
import time
import matplotlib.pyplot as plt

# Initialize Taichi on GPU
ti.init(arch=ti.gpu)  

# Constants
dtype = np.float32
nz, nx, ny = 100, 100, 100
niters = 400000

# Lattice velocity directions
ex_host = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0, -1, 0, 0, -1, -1, -1, -1, 0, 0], dtype=dtype)
ey_host = np.array([0, 0, 1, 0, 0, 0, 1, -1, 1, -1, 0, -1, 0, 0, 0, -1, 1, -1, 1], dtype=dtype)
ez_host = np.array([0, 0, 0, 1, 1, -1, 0, 0, 1, 1, 0, 0, -1, -1, 1, 0, 0, -1, -1], dtype=dtype)
es_host = (1/3)**0.5
w_host = np.array([1/3, 1/18, 1/18, 1/18, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/18, 1/18, 1/18, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36], dtype=dtype)

# Taichi fields (GPU Memory)
ex = ti.ndarray(dtype=ti.f32, shape=19)
ey = ti.ndarray(dtype=ti.f32, shape=19)
ez = ti.ndarray(dtype=ti.f32, shape=19)
w = ti.ndarray(dtype=ti.f32, shape=19)

rho = ti.ndarray(dtype=ti.f32, shape=(nz, ny, nx))
tau = ti.ndarray(dtype=ti.f32, shape=(nz, ny, nx))
u = ti.ndarray(dtype=ti.f32, shape=(2, nz, ny, nx))
Fg = ti.ndarray(dtype=ti.f32, shape=(2, nz, ny, nx))
nodetype = ti.ndarray(dtype=ti.i32, shape=(nz, ny, nx))
f = ti.ndarray(dtype=ti.f32, shape=(19, nz, ny, nx))
#f_old = ti.field(dtype=ti.f32, shape=(ny, nx, 9))

# Copy constant data to GPU fields
ex.from_numpy(ex_host)
ey.from_numpy(ey_host)
ez.from_numpy(ez_host)
w.from_numpy(w_host)

@ti.kernel
def compute_macro_vars_gpu(nodetype: ti.types.ndarray(), rho: ti.types.ndarray(), u: ti.types.ndarray(), f: ti.types.ndarray()):
    for i, j, k in ti.ndrange(nz, ny, nx):
        if nodetype.get_at(i, j, k) <= 0:
            rho_ijk = sum(f.get_at(q, i, j, k) for q in range(19))
            fdotex = sum(f.get_at(q, i, j, k) for q in [1, 4, 5, 6, 7]) - sum(f.get_at(q, i, j, k) for q in [10, 13, 14, 15, 16])
            fdotey = sum(f.get_at(q, i, j, k) for q in [2, 6, 8, 16, 18]) - sum(f.get_at(q, i, j, k) for q in [7, 9, 11, 15, 17])
            fdotez = sum(f.get_at(q, i, j, k) for q in [3, 4, 8, 9, 14]) - sum(f.get_at(q, i, j, k) for q in [5, 12, 13, 17, 18])
            
            rho.set_at(i, j, k, rho_ijk)
            u.set_at(0, i, j, k, fdotex / rho_ijk)
            u.set_at(1, i, j, k, fdotey / rho_ijk)
            u.set_at(2, i, j, k, fdotez / rho_ijk)
        else:
            rho.set_at(i, j, k, 0)
            u.set_at(0, i, j, k, 0)
            u.set_at(1, i, j, k, 0)
            u.set_at(2, i, j, k, 0)


@ti.kernel
def collide_gpu(nodetype: ti.types.ndarray(), rho: ti.types.ndarray(), tau: ti.types.ndarray(), u: ti.types.ndarray(), Fg: ti.types.ndarray(), f: ti.types.ndarray()):
    for i, j, k in ti.ndrange(nz, ny, nx):
        if nodetype.get_at(i, j, k) <= 0:
            u.set_at(0, i, j, k, u.get_at(0, i, j, k) + Fg.get_at(0, i, j, k) * tau.get_at(i, j, k) / rho.get_at(i, j, k))
            u.set_at(1, i, j, k, u.get_at(1, i, j, k) + Fg.get_at(1, i, j, k) * tau.get_at(i, j, k) / rho.get_at(i, j, k))
            u.set_at(2, i, j, k, u.get_at(2, i, j, k) + Fg.get_at(2, i, j, k) * tau.get_at(i, j, k) / rho.get_at(i, j, k))

            # Compute equilibrium distribution function explicitly
            feq0  = rho.get_at(i, j, k) * (-1.0/2.0 * u.get_at(0, i, j, k)**2 - 1.0/2.0 * u.get_at(1, i, j, k)**2 - 1.0/2.0 * u.get_at(2, i, j, k)**2 + 1.0/3.0)
            feq1  = rho.get_at(i, j, k) * ((1.0/6.0) * u.get_at(0, i, j, k)**2 + (1.0/6.0) * u.get_at(0, i, j, k) - 1.0/12.0 * u.get_at(1, i, j, k)**2 - 1.0/12.0 * u.get_at(2, i, j, k)**2 + 1.0/18.0)
            feq2  = rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k)**2 + (1.0/6.0) * u.get_at(1, i, j, k)**2 + (1.0/6.0) * u.get_at(1, i, j, k) - 1.0/12.0 * u.get_at(2, i, j, k)**2 + 1.0/18.0)
            feq3  = rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k)**2 - 1.0/12.0 * u.get_at(1, i, j, k)**2 + (1.0/6.0) * u.get_at(2, i, j, k)**2 + (1.0/6.0) * u.get_at(2, i, j, k) + 1.0/18.0)
            feq4  = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 + (1.0/12.0) * u.get_at(0, i, j, k) - 1.0/24.0 * u.get_at(1, i, j, k)**2 - 1.0/24.0 * u.get_at(2, i, j, k)**2 + (1.0/12.0) * u.get_at(2, i, j, k) + (1.0/8.0) * (u.get_at(0, i, j, k) + u.get_at(2, i, j, k))**2 + 1.0/36.0)
            feq5  = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 + (1.0/12.0) * u.get_at(0, i, j, k) - 1.0/24.0 * u.get_at(1, i, j, k)**2 - 1.0/24.0 * u.get_at(2, i, j, k)**2 - 1.0/12.0 * u.get_at(2, i, j, k) + (1.0/8.0) * (u.get_at(0, i, j, k) - u.get_at(2, i, j, k))**2 + 1.0/36.0)
            feq6  = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 + (1.0/12.0) * u.get_at(0, i, j, k) - 1.0/24.0 * u.get_at(1, i, j, k)**2 + (1.0/12.0) * u.get_at(1, i, j, k) - 1.0/24.0 * u.get_at(2, i, j, k)**2 + (1.0/8.0) * (u.get_at(0, i, j, k) + u.get_at(1, i, j, k))**2 + 1.0/36.0)
            feq7  = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 + (1.0/12.0) * u.get_at(0, i, j, k) - 1.0/24.0 * u.get_at(1, i, j, k)**2 - 1.0/12.0 * u.get_at(1, i, j, k) - 1.0/24.0 * u.get_at(2, i, j, k)**2 + (1.0/8.0) * (u.get_at(0, i, j, k) - u.get_at(1, i, j, k))**2 + 1.0/36.0)
            feq8  = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 - 1.0/24.0 * u.get_at(1, i, j, k)**2 + (1.0/12.0) * u.get_at(1, i, j, k) - 1.0/24.0 * u.get_at(2, i, j, k)**2 + (1.0/12.0) * u.get_at(2, i, j, k) + (1.0/8.0) * (u.get_at(1, i, j, k) + u.get_at(2, i, j, k))**2 + 1.0/36.0)
            feq9  = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 - 1.0/24.0 * u.get_at(1, i, j, k)**2 - 1.0/12.0 * u.get_at(1, i, j, k) - 1.0/24.0 * u.get_at(2, i, j, k)**2 + (1.0/12.0) * u.get_at(2, i, j, k) + (1.0/8.0) * (-u.get_at(1, i, j, k) + u.get_at(2, i, j, k))**2 + 1.0/36.0)
            feq10 = rho.get_at(i, j, k) * ((1.0/6.0) * u.get_at(0, i, j, k)**2 - 1.0/6.0 * u.get_at(0, i, j, k) - 1.0/12.0 * u.get_at(1, i, j, k)**2 - 1.0/12.0 * u.get_at(2, i, j, k)**2 + 1.0/18.0)
            feq11 = rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k)**2 + (1.0/6.0) * u.get_at(1, i, j, k)**2 - 1.0/6.0 * u.get_at(1, i, j, k) - 1.0/12.0 * u.get_at(2, i, j, k)**2 + 1.0/18.0)
            feq12 = rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k)**2 - 1.0/12.0 * u.get_at(1, i, j, k)**2 + (1.0/6.0) * u.get_at(2, i, j, k)**2 - 1.0/6.0 * u.get_at(2, i, j, k) + 1.0/18.0)
            feq13 = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 - 1.0/12.0 * u.get_at(0, i, j, k) - 1.0/24.0 * u.get_at(1, i, j, k)**2 - 1.0/24.0 * u.get_at(2, i, j, k)**2 - 1.0/12.0 * u.get_at(2, i, j, k) + (1.0/8.0) * (-u.get_at(0, i, j, k) - u.get_at(2, i, j, k))**2 + 1.0/36.0)
            feq14 = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 - 1.0/12.0 * u.get_at(0, i, j, k) - 1.0/24.0 * u.get_at(1, i, j, k)**2 - 1.0/24.0 * u.get_at(2, i, j, k)**2 + (1.0/12.0) * u.get_at(2, i, j, k) + (1.0/8.0) * (-u.get_at(0, i, j, k) + u.get_at(2, i, j, k))**2 + 1.0/36.0)
            feq15 = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 - 1.0/12.0 * u.get_at(0, i, j, k) - 1.0/24.0 * u.get_at(1, i, j, k)**2 - 1.0/12.0 * u.get_at(1, i, j, k) - 1.0/24.0 * u.get_at(2, i, j, k)**2 + (1.0/8.0) * (-u.get_at(0, i, j, k) - u.get_at(1, i, j, k))**2 + 1.0/36.0)
            feq16 = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 - 1.0/12.0 * u.get_at(0, i, j, k) - 1.0/24.0 * u.get_at(1, i, j, k)**2 + (1.0/12.0) * u.get_at(1, i, j, k) - 1.0/24.0 * u.get_at(2, i, j, k)**2 + (1.0/8.0) * (-u.get_at(0, i, j, k) + u.get_at(1, i, j, k))**2 + 1.0/36.0)
            feq17 = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 - 1.0/24.0 * u.get_at(1, i, j, k)**2 - 1.0/12.0 * u.get_at(1, i, j, k) - 1.0/24.0 * u.get_at(2, i, j, k)**2 - 1.0/12.0 * u.get_at(2, i, j, k) + (1.0/8.0) * (-u.get_at(1, i, j, k) - u.get_at(2, i, j, k))**2 + 1.0/36.0)
            feq18 = rho.get_at(i, j, k) * (-1.0/24.0 * u.get_at(0, i, j, k)**2 - 1.0/24.0 * u.get_at(1, i, j, k)**2 + (1.0/12.0) * u.get_at(1, i, j, k) - 1.0/24.0 * u.get_at(2, i, j, k)**2 - 1.0/12.0 * u.get_at(2, i, j, k) + (1.0/8.0) * (u.get_at(1, i, j, k) - u.get_at(2, i, j, k))**2 + 1.0/36.0)


            # Relaxation step
            f.set_at(0, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(0, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq0)
            f.set_at(1, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(1, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq1)
            f.set_at(2, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(2, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq2)
            f.set_at(3, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(3, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq3)
            f.set_at(4, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(4, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq4)
            f.set_at(5, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(5, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq5)
            f.set_at(6, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(6, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq6)
            f.set_at(7, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(7, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq7)
            f.set_at(8, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(8, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq8)
            f.set_at(9, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(9, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq9)
            f.set_at(10, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(10, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq10)
            f.set_at(11, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(11, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq11)
            f.set_at(12, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(12, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq12)
            f.set_at(13, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(13, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq13)
            f.set_at(14, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(14, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq14)
            f.set_at(15, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(15, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq15)
            f.set_at(16, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(16, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq16)
            f.set_at(17, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(17, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq17)
            f.set_at(18, i, j, k, (1.0 - (1.0 / tau.get_at(i, j, k))) * f.get_at(18, i, j, k) + (1.0 / tau.get_at(i, j, k)) * feq18)

            for q in range(1, 10):
                fswap = f.get_at(q, i, j, k)
                f.set_at(q, i, j, k, f.get_at(q+9, i, j, k))
                f.set_at(q+9, i, j, k, fswap)


      

@ti.kernel
def compute_edf_gpu(nodetype: ti.types.ndarray(), rho: ti.types.ndarray(), u: ti.types.ndarray(), f: ti.types.ndarray()):
    for i, j, k in ti.ndrange(nz, ny, nx):
        if nodetype.get_at(i, j, k) <= 0:
            f.set_at(0, i, j, k, rho.get_at(i, j, k) * (-1.0/2.0 * u.get_at(0, i, j, k)**2 - 1.0/2.0 * u.get_at(1, i, j, k)**2 - 1.0/2.0 * u.get_at(2, i, j, k)**2 + 1.0/3.0))
            f.set_at(1, i, j, k, rho.get_at(i, j, k) * ((1.0/6.0) * u.get_at(0, i, j, k)**2 + (1.0/6.0) * u.get_at(0, i, j, k) - 1.0/12.0 * u.get_at(1, i, j, k)**2 - 1.0/12.0 * u.get_at(2, i, j, k)**2 + 1.0/18.0))
            f.set_at(2, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k)**2 + (1.0/6.0) * u.get_at(1, i, j, k)**2 + (1.0/6.0) * u.get_at(1, i, j, k) - 1.0/12.0 * u.get_at(2, i, j, k)**2 + 1.0/18.0))
            f.set_at(3, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k)**2 - 1.0/12.0 * u.get_at(1, i, j, k)**2 + (1.0/6.0) * u.get_at(2, i, j, k)**2 + (1.0/6.0) * u.get_at(2, i, j, k) + 1.0/18.0))
            f.set_at(4, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k) * u.get_at(2, i, j, k) + 1.0/36.0))
            f.set_at(5, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k) * u.get_at(2, i, j, k) + 1.0/36.0))
            f.set_at(6, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k) * u.get_at(1, i, j, k) + 1.0/36.0))
            f.set_at(7, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k) * u.get_at(1, i, j, k) + 1.0/36.0))
            f.set_at(8, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(1, i, j, k) * u.get_at(2, i, j, k) + 1.0/36.0))
            f.set_at(9, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(1, i, j, k) * u.get_at(2, i, j, k) + 1.0/36.0))
            f.set_at(10, i, j, k, rho.get_at(i, j, k) * ((1.0/6.0) * u.get_at(0, i, j, k)**2 - 1.0/6.0 * u.get_at(0, i, j, k) + 1.0/18.0))
            f.set_at(11, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k)**2 + (1.0/6.0) * u.get_at(1, i, j, k)**2 - 1.0/6.0 * u.get_at(1, i, j, k) + 1.0/18.0))
            f.set_at(12, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k)**2 + (1.0/6.0) * u.get_at(2, i, j, k)**2 - 1.0/6.0 * u.get_at(2, i, j, k) + 1.0/18.0))
            f.set_at(13, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k) * u.get_at(2, i, j, k) + 1.0/36.0))
            f.set_at(14, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k) * u.get_at(2, i, j, k) + 1.0/36.0))
            f.set_at(15, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k) * u.get_at(1, i, j, k) + 1.0/36.0))
            f.set_at(16, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(0, i, j, k) * u.get_at(1, i, j, k) + 1.0/36.0))
            f.set_at(17, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(1, i, j, k) * u.get_at(2, i, j, k) + 1.0/36.0))
            f.set_at(18, i, j, k, rho.get_at(i, j, k) * (-1.0/12.0 * u.get_at(1, i, j, k) * u.get_at(2, i, j, k) + 1.0/36.0))

# @ti.kernel
# def collide_gpu():
#     for i, j, k in ti.ndrange(nz, ny, nx):
#         if nodetype[i, j, k] <= 0:
#             u[0, i, j, k] += Fg[0, i, j, k] * tau[i, j, k] / rho[i, j, k]
#             u[1, i, j, k] += Fg[1, i, j, k] * tau[i, j, k] / rho[i, j, k]
#             u[2, i, j, k] += Fg[2, i, j, k] * tau[i, j, k] / rho[i, j, k]

#             # Compute equilibrium distribution function explicitly
#             feq0= rho[i, j, k]*(-1.0/2.0*u[0,i, j, k]**2 - 1.0/2.0*u[1,i, j, k]**2 - 1.0/2.0*u[2,i, j, k]**2 + 1.0/3.0)
#             feq1= rho[i, j, k]*((1.0/6.0)*u[0, i, j, k]**2 + (1.0/6.0)*u[0, i, j, k] - 1.0/12.0*u[1, i, j, k]**2 - 1.0/12.0*u[2, i, j, k]**2 + 1.0/18.0)
#             feq2= rho[i, j, k]*(-1.0/12.0*u[0, i, j, k]**2 + (1.0/6.0)*u[1, i, j, k]**2 + (1.0/6.0)*u[1, i, j, k] - 1.0/12.0*u[2, i, j, k]**2 + 1.0/18.0)
#             feq3 = rho[i, j, k]*(-1.0/12.0*u[0, i, j, k]**2 - 1.0/12.0*u[1, i, j, k]**2 + (1.0/6.0)*u[2, i, j, k]**2 + (1.0/6.0)*u[2, i, j, k] + 1.0/18.0)
#             feq4= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 + (1.0/12.0)*u[0, i, j, k] - 1.0/24.0*u[1, i, j, k]**2 - 1.0/24.0*u[2, i, j, k]**2 + (1.0/12.0)*u[2, i, j, k] + (1.0/8.0)*(u[0, i, j, k] + u[2, i, j, k])**2 + 1.0/36.0)
#             feq5= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 + (1.0/12.0)*u[0, i, j, k] - 1.0/24.0*u[1, i, j, k]**2 - 1.0/24.0*u[2, i, j, k]**2 - 1.0/12.0*u[2, i, j, k] + (1.0/8.0)*(u[0, i, j, k] - u[2, i, j, k] )**2 + 1.0/36.0)
#             feq6= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 + (1.0/12.0)*u[0, i, j, k] - 1.0/24.0*u[1, i, j, k]**2 + (1.0/12.0)*u[1, i, j, k] - 1.0/24.0*u[2, i, j, k]**2 + (1.0/8.0)*(u[0, i, j, k] + u[1, i, j, k])**2 + 1.0/36.0)
#             feq7= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 + (1.0/12.0)*u[0, i, j, k] - 1.0/24.0*u[1, i, j, k]**2 - 1.0/12.0*u[1, i, j, k] - 1.0/24.0*u[2, i, j, k]**2 + (1.0/8.0)*(u[0, i, j, k] - u[1, i, j, k])**2 + 1.0/36.0)
#             feq8= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 - 1.0/24.0*u[1, i, j, k]**2 + (1.0/12.0)*u[1, i, j, k] - 1.0/24.0*u[2, i, j, k]**2 + (1.0/12.0)*u[2, i, j, k] + (1.0/8.0)*(u[1, i, j, k] + u[2, i, j, k])**2 + 1.0/36.0)
#             feq9= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 - 1.0/24.0*u[1, i, j, k]**2 - 1.0/12.0*u[1, i, j, k] - 1.0/24.0*u[2, i, j, k]**2 + (1.0/12.0)*u[2, i, j, k] + (1.0/8.0)*(-u[1, i, j, k] + u[2, i, j, k])**2 + 1.0/36.0)
#             feq10= rho[i, j, k]*((1.0/6.0)*u[0, i, j, k]**2 - 1.0/6.0*u[0, i, j, k] - 1.0/12.0*u[1, i, j, k]**2 - 1.0/12.0*u[2, i, j, k]**2 + 1.0/18.0)
#             feq11= rho[i, j, k]*(-1.0/12.0*u[0, i, j, k]**2 + (1.0/6.0)*u[1, i, j, k]**2 - 1.0/6.0*u[1, i, j, k] - 1.0/12.0*u[2, i, j, k]**2 + 1.0/18.0)
#             feq12= rho[i, j, k]*(-1.0/12.0*u[0, i, j, k]**2 - 1.0/12.0*u[1, i, j, k]**2 + (1.0/6.0)*u[2, i, j, k]**2 - 1.0/6.0*u[2, i, j, k] + 1.0/18.0)
#             feq13= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 - 1.0/12.0*u[0, i, j, k] - 1.0/24.0*u[1, i, j, k]**2 - 1.0/24.0*u[2, i, j, k]**2 - 1.0/12.0*u[2, i, j, k] + (1.0/8.0)*(-u[0, i, j, k] - u[2, i, j, k])**2 + 1.0/36.0)
#             feq14= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 - 1.0/12.0*u[0, i, j, k] - 1.0/24.0*u[1, i, j, k]**2 - 1.0/24.0*u[2, i, j, k]**2 + (1.0/12.0)*u[2, i, j, k] + (1.0/8.0)*(-u[0, i, j, k] + u[2, i, j, k])**2 + 1.0/36.0)
#             feq15= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 - 1.0/12.0*u[0, i, j, k] - 1.0/24.0*u[1, i, j, k]**2 - 1.0/12.0*u[1, i, j, k] - 1.0/24.0*u[2, i, j, k]**2 + (1.0/8.0)*(-u[0, i, j, k] - u[1, i, j, k])**2 + 1.0/36.0)
#             feq16= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 - 1.0/12.0*u[0, i, j, k] - 1.0/24.0*u[1, i, j, k]**2 + (1.0/12.0)*u[1, i, j, k] - 1.0/24.0*u[2, i, j, k]**2 + (1.0/8.0)*(-u[0, i, j, k] + u[1, i, j, k])**2 + 1.0/36.0)
#             feq17= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 - 1.0/24.0*u[1, i, j, k]**2 - 1.0/12.0*u[1, i, j, k] - 1.0/24.0*u[2, i, j, k]**2 - 1.0/12.0*u[2, i, j, k] + (1.0/8.0)*(-u[1, i, j, k] - u[2, i, j, k])**2 + 1.0/36.0)
#             feq18= rho[i, j, k]*(-1.0/24.0*u[0, i, j, k]**2 - 1.0/24.0*u[1, i, j, k]**2 + (1.0/12.0)*u[1, i, j, k] - 1.0/24.0*u[2, i, j, k]**2 - 1.0/12.0*u[2, i, j, k] + (1.0/8.0)*(u[1, i, j, k] - u[2, i, j, k])**2 + 1.0/36.0)


#             # Collision step
#             f[0, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[0, i, j, k] + (1.0 / tau[i, j, k]) * feq0
#             f[1, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[1, i, j, k] + (1.0 / tau[i, j, k]) * feq1
#             f[2, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[2, i, j, k] + (1.0 / tau[i, j, k]) * feq2
#             f[3, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[3, i, j, k] + (1.0 / tau[i, j, k]) * feq3
#             f[4, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[4, i, j, k] + (1.0 / tau[i, j, k]) * feq4
#             f[5, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[5, i, j, k] + (1.0 / tau[i, j, k]) * feq5
#             f[6, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[6, i, j, k] + (1.0 / tau[i, j, k]) * feq6
#             f[7, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[7, i, j, k] + (1.0 / tau[i, j, k]) * feq7
#             f[8, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[8, i, j, k] + (1.0 / tau[i, j, k]) * feq8
#             f[9, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[9, i, j, k] + (1.0 / tau[i, j, k]) * feq9
#             f[10, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[10, i, j, k] + (1.0 / tau[i, j, k]) * feq10
#             f[11, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[11, i, j, k] + (1.0 / tau[i, j, k]) * feq11
            # f[12, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[12, i, j, k] + (1.0 / tau[i, j, k]) * feq12
            # f[13, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[13, i, j, k] + (1.0 / tau[i, j, k]) * feq13
            # f[14, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[14, i, j, k] + (1.0 / tau[i, j, k]) * feq14
            # f[15, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[15, i, j, k] + (1.0 / tau[i, j, k]) * feq15
            # f[16, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[16, i, j, k] + (1.0 / tau[i, j, k]) * feq16
            # f[17, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[17, i, j, k] + (1.0 / tau[i, j, k]) * feq17
            # f[18, i, j, k] = (1.0 - (1.0 / tau[i, j, k])) * f[18, i, j, k] + (1.0 / tau[i, j, k]) * feq18

            # for q in range(1,10):
            #     fswap = f[q, i, j, k]
            #     f[q, i, j, k]=f[q+9,i, j, k]
            #     f[q+9,i, j, k]=fswap


# @ti.kernel
# def update_f_old():
#     for i, j, q in ti.ndrange(ny, nx, 9):
#         f_old[i, j, q] = f[i, j, q]

@ti.kernel
def stream_and_bounce_gpu(nodetype: ti.types.ndarray(), f: ti.types.ndarray(), ex: ti.types.ndarray(), ey: ti.types.ndarray(), ez: ti.types.ndarray()):
    for i, j, k in ti.ndrange(nz, ny, nx):
        if nodetype.get_at(i, j, k) <= 0:
            for q in range(1, 10):
                nexti = ti.cast(i - ez.get_at(q), ti.i32)  # Explicit integer casting
                nextj = ti.cast(j - ey.get_at(q), ti.i32)
                nextk = ti.cast(k + ex.get_at(q), ti.i32)

                # Apply periodic boundary conditions efficiently
                nexti = ti.select(nexti >= nz, 0, nexti)
                nextj = ti.select(nextj >= ny, 0, nextj)
                nextk = ti.select(nextk >= nx, 0, nextk)

                if nodetype.get_at(nexti, nextj, nextk) <= 0:
                    # Perform streaming swap
                    temp_f = f.get_at(q + 9, i, j, k)
                    f.set_at(q + 9, i, j, k, f.get_at(q, nexti, nextj, nextk))
                    f.set_at(q, nexti, nextj, nextk, temp_f)

def test_lb():
    # Initialize Fields
    rho_np = np.ones((nz, ny, nx), dtype=dtype)
    tau_np = np.ones((nz, ny, nx), dtype=dtype)
    u_np = np.zeros((2, nz, ny, nx), dtype=dtype)
    Fg_np = np.zeros((2, nz, ny, nx), dtype=dtype)
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

    compute_edf_gpu(nodetype, rho, u, f)

    # Run Simulation
    t0 = time.time()
    for i in range(niters):
        collide_gpu(nodetype, rho, tau, u, Fg, f)
        #update_f_old()
        stream_and_bounce_gpu(nodetype, f, ex, ey, ez)
        compute_macro_vars_gpu(nodetype, rho, u, f)
    t1 = time.time()

    # Copy results back
    u_np = u.to_numpy()
    f_np = f.to_numpy()

    mlups = (nz, ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken:", t1 - t0)

    plt.figure()
    plt.plot(u_np[0][:, :, int(nx / 2)])
    plt.savefig("profile_taichi_swap_unroll.png", dpi=300)

test_lb()
