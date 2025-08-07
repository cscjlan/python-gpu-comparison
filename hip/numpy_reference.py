import numpy as np
import time
import matplotlib.pyplot as plt
import sys

# Constants
dtype = np.float32
nx, ny = 20, 10
niters = 400

# Lattice velocity directions
ex = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
es = (1/3)**0.5
w = np.array([4/9, 1/9, 1/9, 1/36, 1/36, 1/9, 1/9, 1/36, 1/36], dtype=dtype)

rho = np.ones((ny, nx), dtype=dtype)
tau = np.ones((ny, nx), dtype=dtype)
u = np.zeros((2, ny, nx), dtype=dtype)
Fg = np.zeros((2, ny, nx), dtype=dtype)
Fg[0, :, :] = 1e-7
nodetype = np.zeros((ny, nx), dtype=np.int32)
nodetype[0, :] = 1
nodetype[-1, :] = 1
f = np.zeros((9, ny, nx), dtype=dtype)

def compute_macro_vars_gpu():
    for i, j in np.ndindex(ny, nx):
        f_ij = np.array([f[0, i, j], f[1, i, j], f[2, i, j], f[3, i, j], f[4, i, j], f[5, i, j], f[6, i, j], f[7, i, j], f[8, i, j]])
        s = float(nodetype[i, j] <= 0)
        rho_ij = f_ij[0] + f_ij[1] + f_ij[2] + f_ij[3] + f_ij[4] + f_ij[5] + f_ij[6] + f_ij[7] + f_ij[8]
        fdotex = f_ij[1] + f_ij[3] + f_ij[4] - f_ij[5] - f_ij[7] - f_ij[8]
        fdotey = f_ij[2] + f_ij[3] - f_ij[4] - f_ij[6] - f_ij[7] + f_ij[8]

        inv_rho = np.min((1.0 / rho_ij, sys.float_info.max))

        rho[i, j] = s * rho_ij
        u[0, i, j] = s * fdotex * inv_rho
        u[1, i, j] = s * fdotey * inv_rho

def compute_edf_gpu():
    for i, j in np.ndindex(ny, nx):
        s = float(nodetype[i, j] <= 0)
        u0 = u[0, i, j]
        u1 = u[1, i, j]
        for q in range(9):
            exq = ex[q]
            eyq = ey[q]
            inv_es_sq = 3.0

            ux2 = u0 * u0
            uy2 = u1 * u1
            euxy = exq * eyq * u0 * u1
            euxx = exq * exq * ux2
            euyy = eyq * eyq * uy2
            eu2 = 2.0 * euxy + euxx + euyy
            u2 = ux2 + uy2

            term1 = inv_es_sq * (exq * u0 + eyq * u1)
            term2 = 0.5 * inv_es_sq * (inv_es_sq * eu2 - u2)
            f_old = f[q, i, j]
            f_new = w[q] * rho[i, j] * (1.0 + term1 + term2)
            f[q, i, j] = s * f_new + (1.0 - s) * f_old

def collide_gpu():
    for i, j in np.ndindex(ny, nx):
        tau_ij = tau[i, j]
        if nodetype[i, j] <= 0:
            u[0, i, j] += Fg[0, i, j] * tau_ij / rho[i, j]
            u[1, i, j] += Fg[1, i, j] * tau_ij / rho[i, j]

            # Compute equilibrium distribution function explicitly
            feq0 = rho[i, j] * (-2.0/3.0  * u[0, i, j]**2 - 2.0/3.0  * u[1, i, j]**2                                                                                           + 4.0/9.0)
            feq1 = rho[i, j] * (-1.0/6.0  * u[1, i, j]**2 + 1.0/3.0  * u[0, i, j]**2 + 1.0/3.0  * u[0, i, j]                                                                   + 1.0/9.0)
            feq5 = rho[i, j] * (-1.0/6.0  * u[1, i, j]**2 + 1.0/3.0  * u[0, i, j]**2 - 1.0/3.0  * u[0, i, j]                                                                   + 1.0/9.0)
            feq2 = rho[i, j] * (-1.0/6.0  * u[0, i, j]**2 + 1.0/3.0  * u[1, i, j]**2 + 1.0/3.0  * u[1, i, j]                                                                   + 1.0/9.0)
            feq6 = rho[i, j] * (-1.0/6.0  * u[0, i, j]**2 + 1.0/3.0  * u[1, i, j]**2 - 1.0/3.0  * u[1, i, j]                                                                   + 1.0/9.0)
            feq3 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/24.0 * u[1, i, j]**2 + 1.0/12.0 * u[0, i, j] + 1.0/12.0 * u[1, i, j] + 1.0/8.0 * ( u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)
            feq4 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/24.0 * u[1, i, j]**2 + 1.0/12.0 * u[0, i, j] - 1.0/12.0 * u[1, i, j] + 1.0/8.0 * ( u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
            feq7 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/24.0 * u[1, i, j]**2 - 1.0/12.0 * u[0, i, j] - 1.0/12.0 * u[1, i, j] + 1.0/8.0 * (-u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
            feq8 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/24.0 * u[1, i, j]**2 - 1.0/12.0 * u[0, i, j] + 1.0/12.0 * u[1, i, j] + 1.0/8.0 * (-u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)

            # Collision step
            f[0, i, j] = (1.0 - (1.0 / tau_ij)) * f[0, i, j] + (1.0 / tau_ij) * feq0
            f[1, i, j] = (1.0 - (1.0 / tau_ij)) * f[1, i, j] + (1.0 / tau_ij) * feq1
            f[2, i, j] = (1.0 - (1.0 / tau_ij)) * f[2, i, j] + (1.0 / tau_ij) * feq2
            f[3, i, j] = (1.0 - (1.0 / tau_ij)) * f[3, i, j] + (1.0 / tau_ij) * feq3
            f[4, i, j] = (1.0 - (1.0 / tau_ij)) * f[4, i, j] + (1.0 / tau_ij) * feq4
            f[5, i, j] = (1.0 - (1.0 / tau_ij)) * f[5, i, j] + (1.0 / tau_ij) * feq5
            f[6, i, j] = (1.0 - (1.0 / tau_ij)) * f[6, i, j] + (1.0 / tau_ij) * feq6
            f[7, i, j] = (1.0 - (1.0 / tau_ij)) * f[7, i, j] + (1.0 / tau_ij) * feq7
            f[8, i, j] = (1.0 - (1.0 / tau_ij)) * f[8, i, j] + (1.0 / tau_ij) * feq8

            for q in range(1,5):
                fswap = f[q, i,j]
                f[q, i,j]=f[q+4,i,j]
                f[q+4,i,j]=fswap

def stream_and_bounce_gpu():
    for i, j in np.ndindex(ny, nx):
        if nodetype[i, j] <= 0:
            for q in range(1,5):
                nexti = int(i-ey[q])
                nextj = int(j+ex[q])
                if nexti > ny-1: nexti = int(0)
                if nextj > nx-1: nextj = int(0)                        
                if nodetype[nexti,nextj]<=0:
                    fswap = f[q,nexti,nextj]
                    f[q,nexti,nextj] = f[q+4,i,j]
                    f[q+4,i,j] = fswap

def test_lb():
    compute_edf_gpu()

    t0 = time.time()
    for i in range(niters):
        collide_gpu()
        stream_and_bounce_gpu()
        compute_macro_vars_gpu()
    t1 = time.time()

    mlups = (ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken:", t1 - t0)

    plt.figure()
    plt.plot(u[0][:, int(nx / 2)])
    plt.savefig("profile_taichi_swap.png", dpi=300)

test_lb()
