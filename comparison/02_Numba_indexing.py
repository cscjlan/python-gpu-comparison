import sys
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import time
import json


@cuda.jit
def compute_macro_vars_gpu(f, nodetype, rho, u):
    j, i = cuda.grid(2)
    ny, nx = nodetype.shape
    if i < ny and j < nx and nodetype[i, j] <= 0:
        rho_ij = (
            f[0, i, j]
            + f[1, i, j]
            + f[2, i, j]
            + f[3, i, j]
            + f[4, i, j]
            + f[5, i, j]
            + f[6, i, j]
            + f[7, i, j]
            + f[8, i, j]
        )
        fdotex = (
            f[1, i, j] + f[3, i, j] + f[4, i, j] - f[5, i, j] - f[7, i, j] - f[8, i, j]
        )
        fdotey = (
            f[2, i, j] + f[3, i, j] - f[4, i, j] - f[6, i, j] - f[7, i, j] + f[8, i, j]
        )

        rho[i, j] = rho_ij
        u[0, i, j] = fdotex / rho_ij
        u[1, i, j] = fdotey / rho_ij
    elif i < ny and j < nx:
        rho[i, j] = 0
        u[0, i, j] = 0
        u[1, i, j] = 0


@cuda.jit
def compute_edf_gpu(rho, u, nodetype, feq, ex, ey, w, es):
    j, i = cuda.grid(2)
    ny, nx = nodetype.shape
    if i < ny and j < nx and nodetype[i, j] <= 0:
        for q in range(9):
            es = (1 / 3) ** 0.5
            Termorder1 = (1.0 / es**2) * (ex[q] * u[0, i, j] + ey[q] * u[1, i, j])
            euxy = ex[q] * ey[q] * u[0, i, j] * u[1, i, j]
            euxx = ex[q] * ex[q] * u[0, i, j] * u[0, i, j]
            euyy = ey[q] * ey[q] * u[1, i, j] * u[1, i, j]
            eu2 = 2 * euxy + euxx + euyy
            ux2 = u[0, i, j] * u[0, i, j]
            uy2 = u[1, i, j] * u[1, i, j]
            u2 = ux2 + uy2
            Termorder2 = (0.5 / es**4) * eu2 - (0.5 / es**2) * u2
            feq[q, i, j] = w[q] * rho[i, j] * (1 + Termorder1 + Termorder2)


@cuda.jit
def collide_gpu(f, rho, u, nodetype, tau, Fg):
    j, i = cuda.grid(2)
    ny, nx = nodetype.shape
    if i < ny and j < nx and nodetype[i, j] <= 0:
        # Apply forcing
        u[0, i, j] += Fg[0, i, j] * tau[i, j] / rho[i, j]
        u[1, i, j] += Fg[1, i, j] * tau[i, j] / rho[i, j]

        # Compute equilibrium distribution function explicitly
        feq0 = rho[i, j] * (
            -2.0 / 3.0 * u[0, i, j] ** 2 - 2.0 / 3.0 * u[1, i, j] ** 2 + 4.0 / 9.0
        )
        feq1 = rho[i, j] * (
            (1.0 / 3.0) * u[0, i, j] ** 2
            + (1.0 / 3.0) * u[0, i, j]
            - 1.0 / 6.0 * u[1, i, j] ** 2
            + 1.0 / 9.0
        )
        feq2 = rho[i, j] * (
            -1.0 / 6.0 * u[0, i, j] ** 2
            + (1.0 / 3.0) * u[1, i, j] ** 2
            + (1.0 / 3.0) * u[1, i, j]
            + 1.0 / 9.0
        )
        feq3 = rho[i, j] * (
            -1.0 / 24.0 * u[0, i, j] ** 2
            + (1.0 / 12.0) * u[0, i, j]
            - 1.0 / 24.0 * u[1, i, j] ** 2
            + (1.0 / 12.0) * u[1, i, j]
            + (1.0 / 8.0) * (u[0, i, j] + u[1, i, j]) ** 2
            + 1.0 / 36.0
        )
        feq4 = rho[i, j] * (
            -1.0 / 24.0 * u[0, i, j] ** 2
            + (1.0 / 12.0) * u[0, i, j]
            - 1.0 / 24.0 * u[1, i, j] ** 2
            - 1.0 / 12.0 * u[1, i, j]
            + (1.0 / 8.0) * (u[0, i, j] - u[1, i, j]) ** 2
            + 1.0 / 36.0
        )
        feq5 = rho[i, j] * (
            (1.0 / 3.0) * u[0, i, j] ** 2
            - 1.0 / 3.0 * u[0, i, j]
            - 1.0 / 6.0 * u[1, i, j] ** 2
            + 1.0 / 9.0
        )
        feq6 = rho[i, j] * (
            -1.0 / 6.0 * u[0, i, j] ** 2
            + (1.0 / 3.0) * u[1, i, j] ** 2
            - 1.0 / 3.0 * u[1, i, j]
            + 1.0 / 9.0
        )
        feq7 = rho[i, j] * (
            -1.0 / 24.0 * u[0, i, j] ** 2
            - 1.0 / 12.0 * u[0, i, j]
            - 1.0 / 24.0 * u[1, i, j] ** 2
            - 1.0 / 12.0 * u[1, i, j]
            + (1.0 / 8.0) * (-u[0, i, j] - u[1, i, j]) ** 2
            + 1.0 / 36.0
        )
        feq8 = rho[i, j] * (
            -1.0 / 24.0 * u[0, i, j] ** 2
            - 1.0 / 12.0 * u[0, i, j]
            - 1.0 / 24.0 * u[1, i, j] ** 2
            + (1.0 / 12.0) * u[1, i, j]
            + (1.0 / 8.0) * (-u[0, i, j] + u[1, i, j]) ** 2
            + 1.0 / 36.0
        )

        # Collision step
        f[0, i, j] = (1.0 - (1.0 / tau[i, j])) * f[0, i, j] + (1.0 / tau[i, j]) * feq0
        f[1, i, j] = (1.0 - (1.0 / tau[i, j])) * f[1, i, j] + (1.0 / tau[i, j]) * feq1
        f[2, i, j] = (1.0 - (1.0 / tau[i, j])) * f[2, i, j] + (1.0 / tau[i, j]) * feq2
        f[3, i, j] = (1.0 - (1.0 / tau[i, j])) * f[3, i, j] + (1.0 / tau[i, j]) * feq3
        f[4, i, j] = (1.0 - (1.0 / tau[i, j])) * f[4, i, j] + (1.0 / tau[i, j]) * feq4
        f[5, i, j] = (1.0 - (1.0 / tau[i, j])) * f[5, i, j] + (1.0 / tau[i, j]) * feq5
        f[6, i, j] = (1.0 - (1.0 / tau[i, j])) * f[6, i, j] + (1.0 / tau[i, j]) * feq6
        f[7, i, j] = (1.0 - (1.0 / tau[i, j])) * f[7, i, j] + (1.0 / tau[i, j]) * feq7
        f[8, i, j] = (1.0 - (1.0 / tau[i, j])) * f[8, i, j] + (1.0 / tau[i, j]) * feq8

        for q in range(1, 5):
            fswap = f[q, i, j]
            f[q, i, j] = f[q + 4, i, j]
            f[q + 4, i, j] = fswap


@cuda.jit
def stream_and_bounce_gpu(f, nodetype, ex, ey):
    j, i = cuda.grid(2)
    ny, nx = nodetype.shape

    if i < ny and j < nx and nodetype[i, j] <= 0:
        for q in range(1, 5):
            nexti = int(i - ey[q])
            nextj = int(j + ex[q])
            if nexti > ny - 1:
                nexti = int(0)
            if nextj > nx - 1:
                nextj = int(0)
            if nodetype[nexti, nextj] <= 0:
                fswap = f[q, nexti, nextj]
                f[q, nexti, nextj] = f[q + 4, i, j]
                f[q + 4, i, j] = fswap


def test_lb():
    if len(sys.argv) < 3:
        print("Give nx and ny as arguments", file=sys.stderr)
        exit(1)

    nx = int(sys.argv[1])
    ny = int(sys.argv[2])

    with open("input.json", "r") as f:
        j = json.load(f)
    niters = j["niters"]

    dtype = np.float32

    rho = np.ones((ny, nx), dtype=dtype) * j["rho"]
    tau = np.ones((ny, nx), dtype=dtype) * j["tau"]
    u = np.zeros((2, ny, nx), dtype=dtype)
    Fg = np.zeros((2, ny, nx), dtype=dtype)
    Fg[0, :, :] = j["fg"]
    nodetype = np.zeros((ny, nx), dtype=dtype)
    nodetype[0, :] = 1
    nodetype[-1, :] = 1
    f = np.zeros((9, ny, nx), dtype=dtype)
    ex_host = j["ex"]
    ey_host = j["ey"]
    es_host = j["es"]
    w_host = j["w"]

    # GPU Memory Allocation
    f_d = cuda.to_device(f)
    rho_d = cuda.to_device(rho)
    u_d = cuda.to_device(u)
    tau_d = cuda.to_device(tau)
    Fg_d = cuda.to_device(Fg)
    nodetype_d = cuda.to_device(nodetype)
    ex = cuda.to_device(ex_host)
    ey = cuda.to_device(ey_host)
    es = cuda.to_device(es_host)
    w = cuda.to_device(w_host)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Run one iteration first to JIT compile & "clear the pipes"
    compute_edf_gpu[blocks_per_grid, threads_per_block](
        rho_d, u_d, nodetype_d, f_d, ex, ey, w, es
    )
    collide_gpu[blocks_per_grid, threads_per_block](
        f_d, rho_d, u_d, nodetype_d, tau_d, Fg_d
    )
    stream_and_bounce_gpu[blocks_per_grid, threads_per_block](f_d, nodetype_d, ex, ey)
    compute_macro_vars_gpu[blocks_per_grid, threads_per_block](
        f_d, nodetype_d, rho_d, u_d
    )

    # Sync before starting timing
    cuda.synchronize()
    t0 = time.time()
    for _ in range(niters - 1):
        collide_gpu[blocks_per_grid, threads_per_block](
            f_d, rho_d, u_d, nodetype_d, tau_d, Fg_d
        )
        stream_and_bounce_gpu[blocks_per_grid, threads_per_block](
            f_d, nodetype_d, ex, ey
        )
        compute_macro_vars_gpu[blocks_per_grid, threads_per_block](
            f_d, nodetype_d, rho_d, u_d
        )
    cuda.synchronize()
    t1 = time.time()

    f = f_d.copy_to_host()
    u = u_d.copy_to_host()

    mlups = (ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken", t1 - t0)

    plt.figure(2)
    plt.plot(u[0][:, int(nx / 2)])
    plt.savefig("profile_numba_indexing_swap", dpi=300)


if __name__ == "__main__":
    test_lb()
