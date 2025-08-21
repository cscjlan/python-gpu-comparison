import numpy as np
import time
import matplotlib.pyplot as plt

# Constants
dtype = np.float32
nx, ny = 500, 500
niters = 400

# Lattice velocity directions
ex = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
w = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 9, 1 / 9, 1 / 36, 1 / 36], dtype=dtype
)

rho = np.ones((ny, nx), dtype=dtype)
tau = np.ones((ny, nx), dtype=dtype)
u = np.zeros((2, ny, nx), dtype=dtype)
Fg = np.zeros((2, ny, nx), dtype=dtype)
Fg[0, :, :] = 0.6
nodetype = np.zeros((ny, nx), dtype=np.int32)
nodetype[0, :] = 1
nodetype[-1, :] = 1
f = np.zeros((9, ny, nx), dtype=dtype)


# Numpy versions using array functions instead of loops
def compute_macro_vars_numpy():
    rho[:] = (nodetype <= 0) * np.tensordot(f, np.ones(9), (0, 0))
    inv_rho = np.clip(1.0 / rho, np.finfo(dtype).min, np.finfo(dtype).max)
    u[:] = (
        (nodetype <= 0)
        * inv_rho
        * np.concatenate(
            (np.tensordot(f, ex, (0, 0)), np.tensordot(f, ey, (0, 0))), axis=0
        ).reshape((u.shape))
    )


def compute_edf_numpy():
    u2 = u * u
    uxy = u[0] * u[1]
    u2_sum = np.sum(u2, axis=0)

    s = nodetype <= 0
    inv_es_sq = 3.0

    euxy = np.outer(ex * ey, uxy).reshape(f.shape)
    euxx = np.outer(ex * ex, u2[0]).reshape(f.shape)
    euyy = np.outer(ey * ey, u2[1]).reshape(f.shape)
    eu2 = 2.0 * euxy + euxx + euyy

    term1 = inv_es_sq * (np.outer(ex, u[0]) + np.outer(ey, u[1])).reshape(f.shape)
    term2 = 0.5 * inv_es_sq * (inv_es_sq * eu2 - u2_sum)
    f_old = f
    f_new = np.outer(w, rho).reshape(f.shape) * (1.0 + term1 + term2)
    f[:] = s * f_new + (1.0 - s) * f_old


def collide_numpy():
    tau_per_rho = np.clip(tau / rho, np.finfo(dtype).min, np.finfo(dtype).max)

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

    multipliers = np.array(
        [
            2.00,
            1.00,
            1.00,
            0.25,
            0.25,
            1.00,
            1.00,
            0.25,
            0.25,
        ]
    )

    f_updated = np.array(
        [
            -u2_sum + 0.33333333,
            u2_sum_m_u2[1] + u[0],
            u2_sum_m_u2[0] + u[1],
            neg_half_u2_sum_p_sum_2 + sum_u,
            neg_half_u2_sum_p_dif_2 + dif_u,
            u2_sum_m_u2[1] - u[0],
            u2_sum_m_u2[0] - u[1],
            neg_half_u2_sum_p_sum_2 - sum_u,
            neg_half_u2_sum_p_dif_2 - dif_u,
        ]
    )

    rho_per_three = rho * 0.3333333333
    inv_tau = np.clip(1.0 / tau, np.finfo(dtype).min, np.finfo(dtype).max)

    # Mapping of indices:
    # 0 <--> 0
    # 1 <--> 5
    # 2 <--> 6
    # 3 <--> 7
    # 4 <--> 8
    q = np.arange(9)
    l = ((q + 3 & 7) + 1) * (q != 0)
    f_eq = np.outer(multipliers[l], rho_per_three).reshape(f.shape) * (
        f_updated + 0.33333333
    )

    f_l = f[l]
    f_new = (1.0 - inv_tau) * f_l + inv_tau * f_eq[l]
    f[q] = (1.0 - s) * f_l + s * f_new


def stream_and_bounce_numpy():
    q, i, j = np.meshgrid(np.arange(1, 5), np.arange(ny), np.arange(nx), indexing="ij")

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
