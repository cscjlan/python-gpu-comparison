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
        rho_ij = rho[i, j]
        inv_tau = 1.0 / tau_ij
        tau_per_rho = np.min((tau_ij / rho_ij, sys.float_info.max))
        s = float(nodetype[i, j] <= 0)
        u0 = u[0, i, j] + s * Fg[0, i, j] * tau_per_rho
        u1 = u[1, i, j] + s * Fg[1, i, j] * tau_per_rho

        u[0, i, j] = u0
        u[1, i, j] = u1

        ux2 = u0 * u0
        uy2 = u1 * u1
        sum_u = u0 + u1
        dif_u = u0 - u1
        sum_2 = sum_u * sum_u
        dif_2 = dif_u * dif_u
        u2 = ux2 + uy2

        multipliers = np.array([
            2.00,
            1.00,
            1.00,
            0.25,
            0.25,
            1.00,
            1.00,
            0.25,
            0.25,
            ])

        f_updated = np.array([
                  -u2 + 0.33333333,
                   u2 - 1.5 * uy2 + u0,   
                   u2 - 1.5 * ux2 + u1,   
            -0.5 * u2 + 1.5 * sum_2 + sum_u,
            -0.5 * u2 + 1.5 * dif_2 + dif_u,
                   u2 - 1.5 * uy2 - u0,
                   u2 - 1.5 * ux2 - u1,
            -0.5 * u2 + 1.5 * sum_2 - sum_u,
            -0.5 * u2 + 1.5 * dif_2 - dif_u,
            ])

        rho_per_three = rho_ij * 0.3333333333
        for q in range(9):
            # Mapping of indices:
            # 0 <--> 0
            # 1 <--> 5
            # 2 <--> 6
            # 3 <--> 7
            # 4 <--> 8
            l = ((q + 3 & 7) + 1) * int(q != 0)

            f_eq = multipliers[l] * rho_per_three * (f_updated[l] + 0.3333333)
            f_lij = f[l, i, j]
            f_new = (1.0 - inv_tau) * f_lij + inv_tau * f_eq
            f[q, i, j] = (1.0 - s) * f_lij + s * f_new

def stream_and_bounce_gpu():
    for i, j in np.ndindex(ny, nx):
        s1 = float(nodetype[i, j] <= 0)
        for q in range(1,5):
            nexti = (ny + int(i - ey[q])) % ny
            nextj = (nx + int(j + ex[q])) % nx

            s2 = float(nodetype[nexti, nextj] <= 0)
            s = s1 * s2
            f1 = f[q, nexti, nextj]
            f2 = f[q + 4, i, j]

            f[q, nexti, nextj] = (1.0 - s) * f1 + s * f2
            f[q + 4, i, j] = (1.0 - s) * f2 + s * f1

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
