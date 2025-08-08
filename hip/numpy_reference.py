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

# GPU friendly versions, but running on CPU
def compute_macro_vars(i: int, j: int):
    f_ij = np.array([f[0, i, j], f[1, i, j], f[2, i, j], f[3, i, j], f[4, i, j], f[5, i, j], f[6, i, j], f[7, i, j], f[8, i, j]])
    s = float(nodetype[i, j] <= 0)
    rho_ij = f_ij[0] + f_ij[1] + f_ij[2] + f_ij[3] + f_ij[4] + f_ij[5] + f_ij[6] + f_ij[7] + f_ij[8]
    fdotex = f_ij[1] + f_ij[3] + f_ij[4] - f_ij[5] - f_ij[7] - f_ij[8]
    fdotey = f_ij[2] + f_ij[3] - f_ij[4] - f_ij[6] - f_ij[7] + f_ij[8]

    inv_rho = np.min((1.0 / rho_ij, sys.float_info.max))

    rho[i, j] = s * rho_ij
    u[0, i, j] = s * fdotex * inv_rho
    u[1, i, j] = s * fdotey * inv_rho

def compute_edf(i: int, j: int):
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

def stream_and_bounce(i: int, j: int):
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

def collide(i: int, j: int):
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

# A single loop calling each of the functions above
def loop(f):
    for i, j in np.ndindex(ny, nx):
        f(i, j)

# Numpy versions using array functions instead of loops
def compute_macro_vars_numpy():
    cx = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1])
    cy = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1])

    rho[:] = (nodetype <= 0) * np.tensordot(f, np.ones(9), (0, 0))
    inv_rho = np.clip(1.0 / rho, 0.0, sys.float_info.max)
    u[0][:] = (nodetype <= 0) * (np.tensordot(f, cx, (0, 0)) * inv_rho)
    u[1][:] = (nodetype <= 0) * (np.tensordot(f, cy, (0, 0)) * inv_rho)

def compute_edf_numpy():
    u0 = u[0]
    u1 = u[1]
    ux2 = u0 * u0
    uy2 = u1 * u1
    uxy = u0 * u1
    u2 = ux2 + uy2

    s = nodetype <= 0
    inv_es_sq = 3.0

    euxy = np.outer(ex * ey, uxy).reshape(f.shape)
    euxx = np.outer(ex * ex, ux2).reshape(f.shape)
    euyy = np.outer(ey * ey, uy2).reshape(f.shape)
    eu2 = 2.0 * euxy + euxx + euyy

    term1 = inv_es_sq * (np.outer(ex, u0) + np.outer(ey, u1)).reshape(f.shape)
    term2 = 0.5 * inv_es_sq * (inv_es_sq * eu2 - u2)
    f_old = f
    f_new = np.outer(w, rho).reshape(f.shape) * (1.0 + term1 + term2)
    f[:] = s * f_new + (1.0 - s) * f_old

def collide_numpy():
    tau_per_rho = np.clip(tau / rho, 0.0, sys.float_info.max)

    s = nodetype <= 0
    u[:] += s * Fg * tau_per_rho

    sum_u = np.sum(u, axis=0)
    dif_u = -np.diff(u, axis=0).reshape(ny, nx)
    sum_2 = 1.5 * sum_u * sum_u
    dif_2 = 1.5 * dif_u * dif_u
    u2 = u * u
    u2_sum = np.sum(u2, axis=0)
    u2_sum_m_u2 = u2_sum - 1.5 * u2
    neg_half_u2_sum = -0.5 * u2_sum
    neg_half_u2_sum_p_sum_2 = neg_half_u2_sum + sum_2
    neg_half_u2_sum_p_dif_2 = neg_half_u2_sum + dif_2

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
        -u2_sum + 0.33333333,
        u2_sum_m_u2[1] + u[0],   
        u2_sum_m_u2[0] + u[1],   
        neg_half_u2_sum_p_sum_2 + sum_u,
        neg_half_u2_sum_p_dif_2 + dif_u,
        u2_sum_m_u2[1] - u[0],
        u2_sum_m_u2[0] - u[1],
        neg_half_u2_sum_p_sum_2 - sum_u,
        neg_half_u2_sum_p_dif_2 - dif_u,
        ])

    rho_per_three = rho * 0.3333333333
    inv_tau = np.clip(1.0 / tau, 0.0, sys.float_info.max)

    # Mapping of indices:
    # 0 <--> 0
    # 1 <--> 5
    # 2 <--> 6
    # 3 <--> 7
    # 4 <--> 8
    q = np.arange(9)
    l = ((q + 3 & 7) + 1) * (q != 0)
    f_eq = np.outer(multipliers[l], rho_per_three).reshape(f.shape) * (f_updated + 0.33333333)

    f_l = f[l]
    f_new = (1.0 - inv_tau) * f_l + inv_tau * f_eq[l]
    f[q] = (1.0 - s) * f_l + s * f_new

def stream_and_bounce_numpy():
    q, i, j = np.meshgrid(np.arange(1, 5), np.arange(ny), np.arange(nx), indexing='ij')

    i = i.flatten()
    j = j.flatten()
    q = q.flatten()

    nexti = ((ny + i - ey[q]) % ny).astype(np.int32)
    nextj = ((nx + j - ex[q]) % nx).astype(np.int32)

    s1 = nodetype[i, j] <= 0
    s2 = nodetype[nexti, nextj] <= 0
    s = s1 * s2

    f1 = f[q, nexti, nextj]
    f2 = f[q + 4, i, j]
    f[q, nexti, nextj] = (1.0 - s) * f1 + s * f2
    f[q + 4, i, j] = (1.0 - s) * f2 + s * f1

def test_lb_numpy():
    compute_edf_numpy()

    t0 = time.time()
    for _ in range(niters):
        collide_numpy()
        stream_and_bounce_numpy()
        compute_macro_vars_numpy()
    t1 = time.time()

    return t1 - t0

def test_lb_loop():
    loop(compute_edf)

    t0 = time.time()
    for _ in range(niters):
        loop(collide)
        loop(stream_and_bounce)
        loop(compute_macro_vars)
    t1 = time.time()

    return t1 - t0

def main():
    elapsed = test_lb_numpy()

    mlups = (ny * nx * niters * 1e-6) / elapsed
    print("MLUPS:", mlups)
    print("Time taken:", elapsed)

    plt.figure()
    plt.plot(u[0][:, int(nx / 2)])
    plt.savefig("profile_taichi_swap.png", dpi=300)

if __name__ == "__main__":
    main()
