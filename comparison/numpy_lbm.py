import numpy as np
from boilerplate.runner import run


class NumpyLBM:
    def initialize(self, host_data):
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

        s = self.nodetype <= 0
        inv_es_sq = 1.0 / (self.es * self.es)

        euxy = np.outer(self.ex * self.ey, uxy).reshape(self.f.shape)
        euxx = np.outer(self.ex * self.ex, u2[0]).reshape(self.f.shape)
        euyy = np.outer(self.ey * self.ey, u2[1]).reshape(self.f.shape)
        eu2 = 2.0 * euxy + euxx + euyy

        term1 = inv_es_sq * (
            np.outer(self.ex, self.u[0]) + np.outer(self.ey, self.u[1])
        ).reshape(self.f.shape)
        term2 = 0.5 * inv_es_sq * (inv_es_sq * eu2 - u2_sum)
        f_old = self.f
        f_new = np.outer(self.w, self.rho).reshape(self.f.shape) * (1.0 + term1 + term2)
        self.f[:] = s * f_new + (1.0 - s) * f_old

    def compute_macro_vars(self):
        self.rho[:] = (self.nodetype <= 0) * np.tensordot(self.f, np.ones(9), (0, 0))
        inv_rho = np.clip(
            1.0 / self.rho, np.finfo(self.f.dtype).min, np.finfo(self.f.dtype).max
        )
        self.u[:] = (
            (self.nodetype <= 0)
            * inv_rho
            * np.concatenate(
                (
                    np.tensordot(self.f, self.ex, (0, 0)),
                    np.tensordot(self.f, self.ey, (0, 0)),
                ),
                axis=0,
            ).reshape((self.u.shape))
        )

    def collide(self):
        tau_per_rho = np.clip(
            self.tau / self.rho, np.finfo(self.f.dtype).min, np.finfo(self.f.dtype).max
        )

        s = self.nodetype <= 0
        self.u[:] += s * self.Fg * tau_per_rho

        sum_u = np.sum(self.u, axis=0)
        dif_u = -np.diff(self.u, axis=0).reshape(self.tau.shape)
        sum_2 = 1.5 * sum_u * sum_u
        dif_2 = 1.5 * dif_u * dif_u
        u2 = self.u * self.u
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
                u2_sum_m_u2[1] + self.u[0],
                u2_sum_m_u2[0] + self.u[1],
                neg_half_u2_sum_p_sum_2 + sum_u,
                neg_half_u2_sum_p_dif_2 + dif_u,
                u2_sum_m_u2[1] - self.u[0],
                u2_sum_m_u2[0] - self.u[1],
                neg_half_u2_sum_p_sum_2 - sum_u,
                neg_half_u2_sum_p_dif_2 - dif_u,
            ]
        )

        rho_per_three = self.rho * 0.3333333333
        inv_tau = np.clip(
            1.0 / self.tau, np.finfo(self.f.dtype).min, np.finfo(self.f.dtype).max
        )

        # Mapping of indices:
        # 0 <--> 0
        # 1 <--> 5
        # 2 <--> 6
        # 3 <--> 7
        # 4 <--> 8
        q = np.arange(9)
        l = ((q + 3 & 7) + 1) * (q != 0)
        f_eq = np.outer(multipliers, rho_per_three).reshape(self.f.shape) * (
            f_updated + 0.33333333
        )

        f_l = self.f[l]
        f_new = (1.0 - inv_tau) * f_l + inv_tau * f_eq[l]
        self.f[q] = (1.0 - s) * f_l + s * f_new

    def stream_and_bounce(self):
        ny = self.tau.shape[0]
        nx = self.tau.shape[1]
        q, i, j = np.meshgrid(
            np.arange(1, 5), np.arange(ny), np.arange(nx), indexing="ij"
        )

        i = i.flatten()
        j = j.flatten()
        q = q.flatten()

        nexti = ((ny + i - self.ey[q]) % ny).astype(np.int32)
        nextj = ((nx + j - self.ex[q]) % nx).astype(np.int32)

        s1 = self.nodetype[i, j] <= 0
        s2 = self.nodetype[nexti, nextj] <= 0
        s = s1 * s2

        f1 = self.f[q, nexti, nextj]
        f2 = self.f[q + 4, i, j]
        self.f[q, nexti, nextj] = (1.0 - s) * f1 + s * f2
        self.f[q + 4, i, j] = (1.0 - s) * f2 + s * f1

    def finish(self):
        pass


if __name__ == "__main__":
    run(NumpyLBM())
