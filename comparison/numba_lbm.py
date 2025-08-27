import sys
import numpy as np
import numba as nb
from numba import cuda
import matplotlib.pyplot as plt
import time
import json

nb_dtype = nb.float32
dtype = np.float32
max_float = np.finfo(dtype).max


@cuda.jit
def compute_edf(rho, u, nodetype, f, ex, ey, w, es):
    tidx, tidy = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    nx, ny = nodetype.shape

    for i in range(tidy, stridey, ny):
        for j in range(tidx, stridex, nx):
            s = nb_dtype(nodetype[i, j] <= 0)
            ux = u[0, i, j]
            uy = u[1, i, j]
            for q in range(9):
                exq = ex[q]
                eyq = ey[q]
                inv_es_sq = 1.0 / (es * es)

                ux2 = ux * ux
                uy2 = uy * uy
                euxy = exq * eyq * ux * uy
                euxx = exq * exq * ux2
                euyy = eyq * eyq * uy2
                eu2 = 2.0 * euxy + euxx + euyy
                u2 = ux2 + uy2

                term1 = inv_es_sq * (exq * ux + eyq * uy)
                term2 = 0.5 * inv_es_sq * (inv_es_sq * eu2 - u2)
                f_old = f[q, i, j]
                f_new = w[q] * rho[i, j] * (1.0 + term1 + term2)
                f[q, i, j] = s * f_new + (1.0 - s) * f_old


@cuda.jit
def compute_macro_vars(f, nodetype, rho, u, ex, ey):
    tidx, tidy = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    nx, ny = nodetype.shape

    for i in range(tidy, stridey, ny):
        for j in range(tidx, stridex, nx):
            s = nb_dtype(nodetype[i, j] <= 0)
            rho_ij = nb_dtype(0.0)
            fdotex = nb_dtype(0.0)
            fdotey = nb_dtype(0.0)

            for q in range(9):
                f_qij = f[q, i, j]
                rho_ij += f_qij
                fdotex += f_qij * ex[q]
                fdotey += f_qij * ey[q]

            inv_rho = min(1.0 / rho_ij, max_float)

            rho[i, j] = s * rho_ij
            u[0, i, j] = s * fdotex * inv_rho
            u[1, i, j] = s * fdotey * inv_rho


@cuda.jit
def stream_and_bounce(f, nodetype, ex, ey):
    tidx, tidy = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    nx, ny = nodetype.shape

    for i in range(tidy, stridey, ny):
        for j in range(tidx, stridex, nx):
            s1 = nb_dtype(nodetype[i, j] <= 0)
            for q in range(1, 5):
                nexti = (ny + int(i - ey[q])) % ny
                nextj = (nx + int(j + ex[q])) % nx

                s2 = nb_dtype(nodetype[nexti, nextj] <= 0)
                s = s1 * s2
                f1 = f[q, nexti, nextj]
                f2 = f[q + 4, i, j]

                f[q, nexti, nextj] = (1.0 - s) * f1 + s * f2
                f[q + 4, i, j] = (1.0 - s) * f2 + s * f1


@cuda.jit
def collide(f, rho, u, nodetype, tau, Fg):
    tidx, tidy = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    nx, ny = nodetype.shape

    for i in range(tidy, stridey, ny):
        for j in range(tidx, stridex, nx):
            s1 = nb_dtype(nodetype[i, j] <= 0)
            s2 = 1.0 - s1
            tau_ij = tau[i, j]
            rho_ij = rho[i, j]
            inv_tau = min(1.0 / tau_ij, max_float)
            third_rho_per_tau = 0.333333 * min(inv_tau * rho_ij, max_float)
            tau_per_rho = min(tau_ij / rho_ij, max_float)

            ux = u[0, i, j] + s1 * Fg[0, i, j] * tau_per_rho
            uy = u[1, i, j] + s1 * Fg[1, i, j] * tau_per_rho

            ux2 = ux * ux
            uy2 = uy * uy
            uxy3 = 3.0 * ux * uy
            ux2_p_ux = ux2 + ux
            ux2_m_ux = ux2 - ux
            uy2_p_uy = uy2 + uy
            uy2_m_uy = uy2 - uy
            half_ux2 = 0.5 * ux2
            half_uy2 = 0.5 * uy2

            multiplier = cuda.local.array(shape=9, dtype=nb_dtype)
            multiplier[0] = 2.00
            multiplier[1] = 1.00
            multiplier[2] = 1.00
            multiplier[3] = 0.25
            multiplier[4] = 0.25
            multiplier[5] = 1.00
            multiplier[6] = 1.00
            multiplier[7] = 0.25
            multiplier[8] = 0.25

            # Compute equilibrium distribution function explicitly
            feq = cuda.local.array(shape=9, dtype=nb_dtype)
            feq[0] = -ux2 - uy2 + 0.333333
            feq[1] = ux2_p_ux - half_uy2
            feq[2] = uy2_p_uy - half_ux2
            feq[3] = ux2_p_ux + uy2_p_uy + uxy3
            feq[4] = ux2_p_ux + uy2_m_uy - uxy3
            feq[5] = ux2_m_ux - half_uy2
            feq[6] = uy2_m_uy - half_ux2
            feq[7] = ux2_m_ux + uy2_m_uy + uxy3
            feq[8] = ux2_m_ux + uy2_p_uy - uxy3

            # Collision step
            for q in range(9):
                f_old = f[q, i, j]
                f_new = (1.0 - inv_tau) * f_old + third_rho_per_tau * multiplier[q] * (
                    feq[q] + 0.3333333
                )
                f[q, i, j] = s2 * f_old + s1 * f_new

            for q in range(1, 5):
                f1 = f[q, i, j]
                f2 = f[q + 4, i, j]
                f[q, i, j] = s2 * f1 + s1 * f2
                f[q + 4, i, j] = s2 * f2 + s1 * f1

            u[0, i, j] = ux
            u[1, i, j] = uy


def main():
    if len(sys.argv) < 4:
        print("Give nx, ny and plot filename as arguments", file=sys.stderr)
        exit(1)

    nx = int(sys.argv[1])
    ny = int(sys.argv[2])

    with open("input.json", "r") as f:
        j = json.load(f)
    niters = j["niters"]

    rho = np.ones((ny, nx), dtype=dtype) * j["rho"]
    tau = np.ones((ny, nx), dtype=dtype) * j["tau"]
    u = np.zeros((2, ny, nx), dtype=dtype)
    Fg = np.zeros((2, ny, nx), dtype=dtype)
    Fg[0, :, :] = j["fg"]
    nodetype = np.zeros((ny, nx), dtype=dtype)
    nodetype[0, :] = 1
    nodetype[-1, :] = 1
    f = np.zeros((9, ny, nx), dtype=dtype)
    ex = j["ex"]
    ey = j["ey"]
    es = j["es"]
    w = j["w"]

    # GPU Memory Allocation
    f_d = cuda.to_device(f)
    rho_d = cuda.to_device(rho)
    u_d = cuda.to_device(u)
    tau_d = cuda.to_device(tau)
    Fg_d = cuda.to_device(Fg)
    nodetype_d = cuda.to_device(nodetype)
    ex_d = cuda.to_device(ex)
    ey_d = cuda.to_device(ey)
    w_d = cuda.to_device(w)

    threads_per_block = (16, 16)
    blocks_per_grid = (64, 64)

    # Run one iteration first to JIT compile & "clear the pipes"
    compute_edf[blocks_per_grid, threads_per_block](
        rho_d, u_d, nodetype_d, f_d, ex_d, ey_d, w_d, es
    )
    collide[blocks_per_grid, threads_per_block](
        f_d, rho_d, u_d, nodetype_d, tau_d, Fg_d
    )
    stream_and_bounce[blocks_per_grid, threads_per_block](f_d, nodetype_d, ex_d, ey_d)
    compute_macro_vars[blocks_per_grid, threads_per_block](
        f_d, nodetype_d, rho_d, u_d, ex_d, ey_d
    )

    u = u_d.copy_to_host()
    if np.any(np.isnan(u)):
        print("Nan in u")
        exit(1)

    # Sync before starting timing
    cuda.synchronize()
    t0 = time.time()
    for _ in range(niters - 1):
        collide[blocks_per_grid, threads_per_block](
            f_d, rho_d, u_d, nodetype_d, tau_d, Fg_d
        )
        stream_and_bounce[blocks_per_grid, threads_per_block](
            f_d, nodetype_d, ex_d, ey_d
        )
        compute_macro_vars[blocks_per_grid, threads_per_block](
            f_d, nodetype_d, rho_d, u_d, ex_d, ey_d
        )
    cuda.synchronize()
    t1 = time.time()

    f = f_d.copy_to_host()
    u = u_d.copy_to_host()

    mlups = (ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken", t1 - t0)

    plt.figure()
    plt.plot(u[0][:, int(nx / 2)])
    plt.savefig(sys.argv[3], dpi=300)


if __name__ == "__main__":
    main()
