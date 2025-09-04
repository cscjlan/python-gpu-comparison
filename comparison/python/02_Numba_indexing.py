from numba import cuda


@cuda.jit
def compute_macro_vars(f, nodetype, rho, u):
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
def compute_edf(rho, u, nodetype, feq, ex, ey, w, es):
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
def collide(f, rho, u, nodetype, tau, Fg):
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
def stream_and_bounce(f, nodetype, ex, ey):
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


class NumbaOriginalLBM:
    def initialize(self, host_data, inputs):
        self.threads_per_block = (16, 16)
        self.blocks_per_grid_x = (
            inputs.nx + self.threads_per_block[0] - 1
        ) // self.threads_per_block[0]
        self.blocks_per_grid_y = (
            inputs.ny + self.threads_per_block[1] - 1
        ) // self.threads_per_block[1]
        self.blocks_per_grid = (self.blocks_per_grid_x, self.blocks_per_grid_y)

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
        )
        stream_and_bounce[self.blocks_per_grid, self.threads_per_block](
            self.f, self.nodetype, self.ex, self.ey
        )
        compute_macro_vars[self.blocks_per_grid, self.threads_per_block](
            self.f,
            self.nodetype,
            self.rho,
            self.u,
        )

    def copy_to_host(self, host_data):
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

    def synchronize(self):
        cuda.synchronize()

    def finish(self):
        pass


if __name__ == "__main__":
    from boilerplate.runner import run

    run(NumbaOriginalLBM())
