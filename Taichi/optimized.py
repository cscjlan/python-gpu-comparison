import numpy as np
import taichi as ti
import time
import matplotlib.pyplot as plt

# Initialize Taichi on GPU
ti.init(arch=ti.gpu)

# Constants
dtype = np.float32
nx, ny = 200, 100
niters = 400
max_f32 = np.finfo(np.float32).max

# Lattice velocity directions
ex_host = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey_host = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
es_host = (1 / 3) ** 0.5
w_host = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 9, 1 / 9, 1 / 36, 1 / 36], dtype=dtype
)

# Taichi fields (GPU Memory)
ex = ti.field(dtype=ti.f32, shape=9)
ey = ti.field(dtype=ti.f32, shape=9)
w = ti.field(dtype=ti.f32, shape=9)

rho = ti.field(dtype=ti.f32, shape=(ny, nx))
tau = ti.field(dtype=ti.f32, shape=(ny, nx))
u = ti.field(dtype=ti.f32, shape=(2, ny, nx))
Fg = ti.field(dtype=ti.f32, shape=(2, ny, nx))
nodetype = ti.field(dtype=ti.i32, shape=(ny, nx))
f = ti.field(dtype=ti.f32, shape=(9, ny, nx))
# f_old = ti.field(dtype=ti.f32, shape=(ny, nx, 9))

# Copy constant data to GPU fields
ex.from_numpy(ex_host)
ey.from_numpy(ey_host)
w.from_numpy(w_host)


# --------------------------------------------------------------
# Old
# --------------------------------------------------------------
@ti.kernel
def compute_macro_vars_old():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
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
                f[1, i, j]
                + f[3, i, j]
                + f[4, i, j]
                - f[5, i, j]
                - f[7, i, j]
                - f[8, i, j]
            )
            fdotey = (
                f[2, i, j]
                + f[3, i, j]
                - f[4, i, j]
                - f[6, i, j]
                - f[7, i, j]
                + f[8, i, j]
            )

            rho[i, j] = rho_ij
            u[0, i, j] = fdotex / rho_ij
            u[1, i, j] = fdotey / rho_ij
        else:
            rho[i, j] = 0
            u[0, i, j] = 0
            u[1, i, j] = 0


@ti.kernel
def compute_edf_old():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
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
                f[q, i, j] = w[q] * rho[i, j] * (1 + Termorder1 + Termorder2)


@ti.kernel
def collide_old():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
            u[0, i, j] += Fg[0, i, j] * tau[i, j] / rho[i, j]
            u[1, i, j] += Fg[1, i, j] * tau[i, j] / rho[i, j]

            # fmt: off
            # Compute equilibrium distribution function explicitly
            feq0 = rho[i, j] * (-2.0/3.0 * u[0, i, j]**2 - 2.0/3.0 * u[1, i, j]**2 + 4.0/9.0)
            feq1 = rho[i, j] * ((1.0/3.0) * u[0, i, j]**2 + (1.0/3.0) * u[0, i, j] - 1.0/6.0 * u[1, i, j]**2 + 1.0/9.0)
            feq2 = rho[i, j] * (-1.0/6.0 * u[0, i, j]**2 + (1.0/3.0) * u[1, i, j]**2 + (1.0/3.0) * u[1, i, j] + 1.0/9.0)
            feq3 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 + (1.0/12.0) * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 + 
                                (1.0/12.0) * u[1, i, j] + (1.0/8.0) * (u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)
            feq4 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 + (1.0/12.0) * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 - 
                                1.0/12.0 * u[1, i, j] + (1.0/8.0) * (u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
            feq5 = rho[i, j] * ((1.0/3.0) * u[0, i, j]**2 - 1.0/3.0 * u[0, i, j] - 1.0/6.0 * u[1, i, j]**2 + 1.0/9.0)
            feq6 = rho[i, j] * (-1.0/6.0 * u[0, i, j]**2 + (1.0/3.0) * u[1, i, j]**2 - 1.0/3.0 * u[1, i, j] + 1.0/9.0)
            feq7 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/12.0 * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 - 
                                1.0/12.0 * u[1, i, j] + (1.0/8.0) * (-u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
            feq8 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/12.0 * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 + 
                                (1.0/12.0) * u[1, i, j] + (1.0/8.0) * (-u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)

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
            # fmt: on

            for q in ti.static(range(1, 5)):
                fswap = f[q, i, j]
                f[q, i, j] = f[q + 4, i, j]
                f[q + 4, i, j] = fswap


@ti.kernel
def stream_and_bounce_old():
    for i, j in ti.ndrange(ny, nx):
        if nodetype[i, j] <= 0:
            for q in ti.static(range(1, 5)):
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


# --------------------------------------------------------------
# New
# --------------------------------------------------------------
@ti.kernel
def compute_macro_vars():
    for i, j in ti.ndrange(ny, nx):
        s = float(nodetype[i, j] <= 0)
        rho_ij = 0.0
        fdotex = 0.0
        fdotey = 0.0

        for q in ti.static(range(9)):
            f_qij = f[q, i, j]
            rho_ij += f_qij
            fdotex += f_qij * ex[q]
            fdotey += f_qij * ey[q]

        inv_rho = ti.min(1.0 / rho_ij, max_f32)

        rho[i, j] = s * rho_ij
        u[0, i, j] = s * fdotex * inv_rho
        u[1, i, j] = s * fdotey * inv_rho


@ti.kernel
def compute_edf():
    for i, j in ti.ndrange(ny, nx):
        s = float(nodetype[i, j] <= 0)
        u0 = u[0, i, j]
        u1 = u[1, i, j]
        for q in ti.static(range(9)):
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


@ti.kernel
def stream_and_bounce():
    for i, j in ti.ndrange(ny, nx):
        s1 = float(nodetype[i, j] <= 0)
        for q in ti.static(range(1, 5)):
            nexti = (ny + int(i - ey[q])) % ny
            nextj = (nx + int(j + ex[q])) % nx

            s2 = float(nodetype[nexti, nextj] <= 0)
            s = s1 * s2
            f1 = f[q, nexti, nextj]
            f2 = f[q + 4, i, j]

            f[q, nexti, nextj] = (1.0 - s) * f1 + s * f2
            f[q + 4, i, j] = (1.0 - s) * f2 + s * f1


@ti.kernel
def collide():
    for i, j in ti.ndrange(ny, nx):
        tau_ij = tau[i, j]
        rho_ij = rho[i, j]
        inv_tau = 1.0 / tau_ij

        tau_per_rho = ti.min(tau_ij / rho_ij, max_f32)
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

        multipliers = ti.static(
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

        f_updated = ti.static(
            [
                -u2 + 0.33333333,
                u2 - 1.5 * uy2 + u0,
                u2 - 1.5 * ux2 + u1,
                -0.5 * u2 + 1.5 * sum_2 + sum_u,
                -0.5 * u2 + 1.5 * dif_2 + dif_u,
                u2 - 1.5 * uy2 - u0,
                u2 - 1.5 * ux2 - u1,
                -0.5 * u2 + 1.5 * sum_2 - sum_u,
                -0.5 * u2 + 1.5 * dif_2 - dif_u,
            ]
        )

        rho_per_three = rho_ij * 0.3333333333
        for q in ti.static(range(9)):
            f_eq = multipliers[q] * rho_per_three * (f_updated[q] + 0.3333333)
            f_qij = f[q, i, j]
            f_new = (1.0 - inv_tau) * f_qij + inv_tau * f_eq
            f[q, i, j] = (1.0 - s) * f_qij + s * f_new

        for q in ti.static(range(1, 5)):
            fswap = f[q, i, j]
            f[q, i, j] = f[q + 4, i, j]
            f[q + 4, i, j] = fswap


@ti.kernel
def collide_updated():
    for i, j in ti.ndrange(ny, nx):
        s = float(nodetype[i, j] <= 0)

        u[0, i, j] += s * Fg[0, i, j] * tau[i, j] / rho[i, j]
        u[1, i, j] += s * Fg[1, i, j] * tau[i, j] / rho[i, j]

        # fmt: off
        # Compute equilibrium distribution function explicitly
        feq0 = rho[i, j] * (-2.0/3.0 * u[0, i, j]**2 - 2.0/3.0 * u[1, i, j]**2 + 4.0/9.0)
        feq1 = rho[i, j] * ((1.0/3.0) * u[0, i, j]**2 + (1.0/3.0) * u[0, i, j] - 1.0/6.0 * u[1, i, j]**2 + 1.0/9.0)
        feq2 = rho[i, j] * (-1.0/6.0 * u[0, i, j]**2 + (1.0/3.0) * u[1, i, j]**2 + (1.0/3.0) * u[1, i, j] + 1.0/9.0)
        feq3 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 + (1.0/12.0) * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 + 
                            (1.0/12.0) * u[1, i, j] + (1.0/8.0) * (u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)
        feq4 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 + (1.0/12.0) * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 - 
                            1.0/12.0 * u[1, i, j] + (1.0/8.0) * (u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
        feq5 = rho[i, j] * ((1.0/3.0) * u[0, i, j]**2 - 1.0/3.0 * u[0, i, j] - 1.0/6.0 * u[1, i, j]**2 + 1.0/9.0)
        feq6 = rho[i, j] * (-1.0/6.0 * u[0, i, j]**2 + (1.0/3.0) * u[1, i, j]**2 - 1.0/3.0 * u[1, i, j] + 1.0/9.0)
        feq7 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/12.0 * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 - 
                            1.0/12.0 * u[1, i, j] + (1.0/8.0) * (-u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
        feq8 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/12.0 * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 + 
                            (1.0/12.0) * u[1, i, j] + (1.0/8.0) * (-u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)

        # Collision step
        f[0, i, j] = (1.0 - s) * f[0, i, j] + s * ((1.0 - (1.0 / tau[i, j])) * f[0, i, j] + (1.0 / tau[i, j]) * feq0)
        f[1, i, j] = (1.0 - s) * f[1, i, j] + s * ((1.0 - (1.0 / tau[i, j])) * f[1, i, j] + (1.0 / tau[i, j]) * feq1)
        f[2, i, j] = (1.0 - s) * f[2, i, j] + s * ((1.0 - (1.0 / tau[i, j])) * f[2, i, j] + (1.0 / tau[i, j]) * feq2)
        f[3, i, j] = (1.0 - s) * f[3, i, j] + s * ((1.0 - (1.0 / tau[i, j])) * f[3, i, j] + (1.0 / tau[i, j]) * feq3)
        f[4, i, j] = (1.0 - s) * f[4, i, j] + s * ((1.0 - (1.0 / tau[i, j])) * f[4, i, j] + (1.0 / tau[i, j]) * feq4)
        f[5, i, j] = (1.0 - s) * f[5, i, j] + s * ((1.0 - (1.0 / tau[i, j])) * f[5, i, j] + (1.0 / tau[i, j]) * feq5)
        f[6, i, j] = (1.0 - s) * f[6, i, j] + s * ((1.0 - (1.0 / tau[i, j])) * f[6, i, j] + (1.0 / tau[i, j]) * feq6)
        f[7, i, j] = (1.0 - s) * f[7, i, j] + s * ((1.0 - (1.0 / tau[i, j])) * f[7, i, j] + (1.0 / tau[i, j]) * feq7)
        f[8, i, j] = (1.0 - s) * f[8, i, j] + s * ((1.0 - (1.0 / tau[i, j])) * f[8, i, j] + (1.0 / tau[i, j]) * feq8)
        # fmt: on

        for q in ti.static(range(1, 5)):
            f1 = f[q, i, j]
            f2 = f[q + 4, i, j]
            f[q, i, j] = (1.0 - s) * f1 + s * f2
            f[q + 4, i, j] = (1.0 - s) * f2 + s * f1


@ti.kernel
def init():
    for i, j in ti.ndrange(ny, nx):
        rho[i, j] = 1.0
        tau[i, j] = 1.0

        u[0, i, j] = 0.0
        u[1, i, j] = 0.0

        Fg[0, i, j] = 1
        Fg[1, i, j] = 0.0

        nodetype[i, j] = int(i == 0) or (i == (ny - 1))

        for q in ti.static(range(9)):
            f[q, i, j] = 0.0


def old():
    init()
    compute_edf_old()
    t0 = time.time()
    for _ in range(niters):
        collide_old()
        stream_and_bounce_old()
        compute_macro_vars_old()
    ti.sync()
    t1 = time.time()

    return t1 - t0


def optimized():
    init()
    compute_edf()
    t0 = time.time()
    for _ in range(niters):
        collide_updated()
        stream_and_bounce()
        compute_macro_vars()
    ti.sync()
    t1 = time.time()

    return t1 - t0


def test_lb(func, name):
    elapsed = func()
    mlups = (ny * nx * niters * 1e-6) / elapsed

    print(f"MLUPS ({name}): {mlups}")
    print(f"Time taken ({name})): {elapsed}")

    u_np = u.to_numpy()

    plt.figure()
    plt.plot(u_np[0][:, int(nx / 2)])
    plt.savefig("profile_taichi_swap_" + name + ".png", dpi=300)


def main():
    test_lb(old, "old")
    test_lb(optimized, "optimized")


if __name__ == "__main__":
    main()
