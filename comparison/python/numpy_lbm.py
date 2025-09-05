import numpy as np
from boilerplate.runner import run
import copy


class NumpyLBM:
    def initialize(self, host_data, _):
        self.f = np.array(host_data.f)
        self.u = np.array(host_data.u)
        self.rho = np.array(host_data.rho)
        self.tau = np.array(host_data.tau)
        self.Fg = np.array(host_data.Fg)
        self.nodetype = np.array(host_data.nodetype)
        self.ex = np.array(host_data.ex)
        self.ey = np.array(host_data.ey)
        self.w = np.array(host_data.w)
        self.es = host_data.es

        self.compute_edf()

    def iterate(self):
        self.collide()
        self.stream_and_bounce()
        self.compute_macro_vars()

    def copy_to_host(self, host_data):
        host_data.f = self.f
        host_data.u = self.u
        host_data.rho = self.rho
        host_data.tau = self.tau
        host_data.Fg = self.Fg
        host_data.nodetype = self.nodetype
        host_data.ex = self.ex
        host_data.ey = self.ey
        host_data.w = self.w

        return host_data

    def synchronize(self):
        pass

    def compute_edf(self):
        u2 = self.u * self.u
        uxy = self.u[0] * self.u[1]
        u2_sum = np.sum(u2, axis=0)

        inv_es_sq = 1.0 / (self.es * self.es)

        euxy = np.outer(self.ex * self.ey, uxy).reshape(self.f.shape)
        euxx = np.outer(self.ex * self.ex, u2[0]).reshape(self.f.shape)
        euyy = np.outer(self.ey * self.ey, u2[1]).reshape(self.f.shape)
        eu2 = 2.0 * euxy + euxx + euyy

        term1 = inv_es_sq * (
            np.outer(self.ex, self.u[0]) + np.outer(self.ey, self.u[1])
        ).reshape(self.f.shape)
        term2 = 0.5 * inv_es_sq * (inv_es_sq * eu2 - u2_sum)
        f_new = np.outer(self.w, self.rho).reshape(self.f.shape) * (1.0 + term1 + term2)
        s = self.nodetype <= 0
        self.f[:, s] = f_new[:, s]

    def compute_macro_vars(self):
        s = self.nodetype <= 0
        self.rho[s] = np.tensordot(self.f, np.ones(9), (0, 0))[s]
        inv_rho = 1.0 / self.rho
        self.u[:, s] = (
            inv_rho
            * np.concatenate(
                (
                    np.tensordot(self.f, self.ex, (0, 0)),
                    np.tensordot(self.f, self.ey, (0, 0)),
                ),
                axis=0,
            ).reshape((self.u.shape))
        )[:, s]

    def collide(self):
        s = self.nodetype <= 0
        self.u[:, s] += (self.Fg * self.tau / self.rho)[:, s]

        u2 = self.u * self.u
        u2_p_u = u2 + self.u
        u2_m_u = u2 - self.u
        uxy3 = 3.0 * self.u[0] * self.u[1]

        rho_per_three = self.rho * 0.3333333333
        inv_tau = np.where(
            self.tau != 0.0,
            1.0 / self.tau,
            np.finfo(self.f.dtype).max,
        )

        one_m_inv_tau = 1.0 - inv_tau

        feq = lambda idx, a, b: (
            one_m_inv_tau * self.f[idx] + inv_tau * a * rho_per_three * (b + 0.333333)
        )

        self.f[0][s] = feq(0, 2.00, -u2[0] - u2[1] + 0.333333)[s]
        self.f[1][s] = feq(1, 1.00, u2_p_u[0] - 0.5 * u2[1])[s]
        self.f[2][s] = feq(2, 1.00, u2_p_u[1] - 0.5 * u2[0])[s]
        self.f[3][s] = feq(3, 0.25, u2_p_u[0] + u2_p_u[1] + uxy3)[s]
        self.f[4][s] = feq(4, 0.25, u2_p_u[0] + u2_m_u[1] - uxy3)[s]
        self.f[5][s] = feq(5, 1.00, u2_m_u[0] - 0.5 * u2[1])[s]
        self.f[6][s] = feq(6, 1.00, u2_m_u[1] - 0.5 * u2[0])[s]
        self.f[7][s] = feq(7, 0.25, u2_m_u[0] + u2_m_u[1] + uxy3)[s]
        self.f[8][s] = feq(8, 0.25, u2_m_u[0] + u2_p_u[1] - uxy3)[s]

        for q in range(1, 5):
            f_copy = copy.deepcopy(self.f[q])
            self.f[q][s] = self.f[q + 4][s]
            self.f[q + 4][s] = f_copy[s]

    def stream_and_bounce(self):
        ny = self.tau.shape[0]
        nx = self.tau.shape[1]
        q, i, j = np.meshgrid(
            np.arange(1, 5),
            np.arange(ny),
            np.arange(nx),
            indexing="ij",
        )

        i = i.flatten()
        j = j.flatten()
        q = q.flatten()

        nexti = ((ny + i - self.ey[q]) % ny).astype(np.int32)
        nextj = ((nx + j + self.ex[q]) % nx).astype(np.int32)

        s = (self.nodetype[i, j] <= 0) & (self.nodetype[nexti, nextj] <= 0)

        f1 = self.f[q, nexti, nextj]
        f2 = self.f[q + 4, i, j]
        self.f[q, nexti, nextj] = (1.0 - s) * f1 + s * f2
        self.f[q + 4, i, j] = (1.0 - s) * f2 + s * f1

    def finish(self):
        pass


if __name__ == "__main__":
    run(NumpyLBM())
