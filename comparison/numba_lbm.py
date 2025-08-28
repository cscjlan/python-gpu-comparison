import sys
import numpy as np
import numba as nb
from numba import cuda
import matplotlib.pyplot as plt
import time
import json


@cuda.jit
def compute_edf(rho, u, nodetype, f, ex, ey, w, es):
    tidx, tidy = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    ny, nx = nodetype.shape

    for i in range(tidy, ny, stridey):
        for j in range(tidx, nx, stridex):
            s = nb.float32(nodetype[i, j] <= 0)
            ux = u[0, i, j]
            uy = u[1, i, j]
            rho_ij = rho[i, j]
            for q in range(9):
                eux = ux * ex[q]
                euy = uy * ey[q]
                inv_es_sq = 1.0 / (es * es)
                eu2 = 2.0 * eux * euy + eux * eux + euy * euy
                term1 = inv_es_sq * (eux + euy)
                term2 = 0.5 * inv_es_sq * (inv_es_sq * eu2 - (ux * ux + uy * uy))
                f_new = w[q] * rho_ij * (1.0 + term1 + term2)
                f_old = f[q, i, j]
                f[q, i, j] = s * f_new + (1.0 - s) * f_old


@cuda.jit
def compute_macro_vars(f, nodetype, rho, u, ex, ey, max_float):
    tidx, tidy = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    ny, nx = nodetype.shape

    for i in range(tidy, ny, stridey):
        for j in range(tidx, nx, stridex):
            s = nb.float32(nodetype[i, j] <= 0)
            rho_ij = nb.float32(0.0)
            fdotex = nb.float32(0.0)
            fdotey = nb.float32(0.0)

            for q in range(9):
                f_qij = f[q, i, j]
                rho_ij += f_qij
                fdotex += f_qij * ex[q]
                fdotey += f_qij * ey[q]

            inv_rho = max_float if rho_ij == 0.0 else 1.0 / rho_ij

            rho[i, j] = s * rho_ij
            u[0, i, j] = s * fdotex * inv_rho
            u[1, i, j] = s * fdotey * inv_rho


@cuda.jit
def stream_and_bounce(f, nodetype, ex, ey):
    tidx, tidy = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    nx, ny = nodetype.shape

    for i in range(tidy, ny, stridey):
        for j in range(tidx, nx, stridex):
            s1 = nb.float32(nodetype[i, j] <= 0)
            for q in range(1, 5):
                nexti = (ny + int(i - ey[q])) % ny
                nextj = (nx + int(j + ex[q])) % nx

                s2 = nb.float32(nodetype[nexti, nextj] <= 0)
                s = s1 * s2
                f1 = f[q, nexti, nextj]
                f2 = f[q + 4, i, j]

                f[q, nexti, nextj] = (1.0 - s) * f1 + s * f2
                f[q + 4, i, j] = (1.0 - s) * f2 + s * f1


# Globals are treated as compile time constants
# maybe it applies to arrays as well?
multiplier = np.array(
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


@cuda.jit
def collide(f, rho, u, nodetype, tau, Fg, max_float):
    tidx, tidy = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    ny, nx = nodetype.shape

    for i in range(tidy, ny, stridey):
        for j in range(tidx, nx, stridex):
            s1 = nb.float32(nodetype[i, j] <= 0)
            s2 = 1.0 - s1
            tau_ij = tau[i, j]
            rho_ij = rho[i, j]
            inv_tau = max_float if tau_ij == 0.0 else 1.0 / tau_ij
            tau_per_rho = max_float if rho_ij == 0.0 else tau_ij / rho_ij
            third_rho_per_tau = 0.333333 * inv_tau * rho_ij

            ux = u[0, i, j] + s1 * Fg[0, i, j] * tau_per_rho
            uy = u[1, i, j] + s1 * Fg[1, i, j] * tau_per_rho

            ux2 = ux * ux
            uy2 = uy * uy
            uxy3 = 3.0 * ux * uy
            ux2_p_ux = ux2 + ux
            ux2_m_ux = ux2 - ux
            uy2_p_uy = uy2 + uy
            uy2_m_uy = uy2 - uy

            # Compute equilibrium distribution function explicitly
            feq = cuda.local.array(shape=9, dtype=f.dtype)
            feq[0] = -ux2 - uy2 + 0.333333
            feq[1] = ux2_p_ux - 0.5 * uy2
            feq[2] = uy2_p_uy - 0.5 * ux2
            feq[3] = ux2_p_ux + uy2_p_uy + uxy3
            feq[4] = ux2_p_ux + uy2_m_uy - uxy3
            feq[5] = ux2_m_ux - 0.5 * uy2
            feq[6] = uy2_m_uy - 0.5 * ux2
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


class Inputs:
    def __init__(self):
        if len(sys.argv) < 5:
            print(
                "Give nx, ny, json input file and plot filename as arguments",
                file=sys.stderr,
            )
            exit(1)

        self.nx = int(sys.argv[1])
        self.ny = int(sys.argv[2])
        self.input_filename = sys.argv[3]
        self.output_filename = sys.argv[4]

        with open(self.input_filename, "r") as f:
            j = json.load(f)

        self.dtype = np.float32 if j["dtype"] == "f32" else np.float64
        self.niters = j["niters"]
        self.ex = j["ex"]
        self.ey = j["ey"]
        self.es = j["es"]
        self.w = j["w"]
        self.Fg = j["Fg"]
        self.rho = j["rho"]
        self.tau = j["tau"]


class HostData:
    def __init__(self, inputs: Inputs):
        shape_1 = (inputs.ny, inputs.nx)
        shape_2 = (2, inputs.ny, inputs.nx)
        shape_9 = (9, inputs.ny, inputs.nx)

        self.rho = np.ones(shape_1, dtype=inputs.dtype) * inputs.rho
        self.tau = np.ones(shape_1, dtype=inputs.dtype) * inputs.tau
        self.u = np.zeros(shape_2, dtype=inputs.dtype)

        self.Fg = np.zeros(shape_2, dtype=inputs.dtype)
        self.Fg[0, :, :] = inputs.Fg

        self.nodetype = np.zeros(shape_1, dtype=inputs.dtype)
        self.nodetype[0, :] = 1
        self.nodetype[-1, :] = 1

        self.f = np.zeros(shape_9, dtype=inputs.dtype)
        self.ex = inputs.ex
        self.ey = inputs.ey
        self.w = inputs.w
        self.es = inputs.es

    def output(self, inputs):
        plt.imsave("u0_" + inputs.output_filename + ".png", self.u[0])
        plt.imsave("u1_" + inputs.output_filename + ".png", self.u[1])


class NumbaLBM:
    def __init__(self, host_data: HostData):
        self.threads_per_block = (32, 16)
        self.blocks_per_grid = (32, 64)
        self.max_float = np.finfo(host_data.rho.dtype).max

        self.f = cuda.to_device(host_data.f)
        self.u = cuda.to_device(host_data.u)
        self.rho = cuda.to_device(host_data.rho)
        self.tau = cuda.to_device(host_data.tau)
        self.Fg = cuda.to_device(host_data.Fg)
        self.nodetype = cuda.to_device(host_data.nodetype)
        self.ex = cuda.to_device(host_data.ex)
        self.ey = cuda.to_device(host_data.ey)
        self.w = cuda.to_device(host_data.w)
        self.es = host_data.es

    def initialize(self):
        compute_edf[self.blocks_per_grid, self.threads_per_block](
            self.rho,
            self.u,
            self.nodetype,
            self.f,
            self.ex,
            self.ey,
            self.w,
            self.es,
        )

    def iterate(self):
        collide[self.blocks_per_grid, self.threads_per_block](
            self.f,
            self.rho,
            self.u,
            self.nodetype,
            self.tau,
            self.Fg,
            self.max_float,
        )
        stream_and_bounce[self.blocks_per_grid, self.threads_per_block](
            self.f, self.nodetype, self.ex, self.ey
        )
        compute_macro_vars[self.blocks_per_grid, self.threads_per_block](
            self.f,
            self.nodetype,
            self.rho,
            self.u,
            self.ex,
            self.ey,
            self.max_float,
        )

    def copy_to_host(self, host_data: HostData):
        host_data.f = self.f.copy_to_host()
        host_data.u = self.u.copy_to_host()
        host_data.rho = self.rho.copy_to_host()
        host_data.tau = self.tau.copy_to_host()
        host_data.Fg = self.Fg.copy_to_host()
        host_data.nodetype = self.nodetype.copy_to_host()
        host_data.ex = self.ex.copy_to_host()
        host_data.ey = self.ey.copy_to_host()
        host_data.w = self.w.copy_to_host()

        return host_data


def run(runner, inputs: Inputs):
    # Initialize and run one iteration to clear the pipes
    runner.initialize()
    runner.iterate()
    cuda.synchronize()

    t0 = time.time()
    for _ in range(inputs.niters - 1):
        runner.iterate()
    cuda.synchronize()
    t1 = time.time()

    elapsed = t1 - t0
    mlups = (inputs.ny * inputs.nx * inputs.niters * 1e-6) / elapsed
    print("MLUPS:", mlups)
    print("Time taken", elapsed)


def main():
    inputs = Inputs()
    host_data = HostData(inputs)

    runner = NumbaLBM(host_data)

    run(runner, inputs)
    host_data = runner.copy_to_host(host_data)
    host_data.output(inputs)


if __name__ == "__main__":
    main()
