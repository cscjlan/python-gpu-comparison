import torch
from boilerplate.runner import run


# TODO: fix the problem with U, try to optimize
class TorchLBM:
    def initialize(self, host_data, _):
        self.device = torch.device("cuda")

        self.f = torch.from_numpy(host_data.f).to(self.device)
        self.u = torch.from_numpy(host_data.u).to(self.device)
        self.rho = torch.from_numpy(host_data.rho).to(self.device)
        self.tau = torch.from_numpy(host_data.tau).to(self.device)
        self.Fg = torch.from_numpy(host_data.Fg).to(self.device)
        self.nodetype = torch.from_numpy(host_data.nodetype).to(self.device)
        self.ex = torch.from_numpy(host_data.ex).to(self.device)
        self.ey = torch.from_numpy(host_data.ey).to(self.device)
        self.w = torch.from_numpy(host_data.w).to(self.device)
        self.es = (torch.ones(()) * host_data.es).to(self.device)

        self.compute_edf()

    def iterate(self):
        self.collide()
        self.stream_and_bounce()
        self.compute_macro_vars()

    def copy_to_host(self, host_data):
        host_data.f = self.f.cpu().numpy()
        host_data.u = self.u.cpu().numpy()
        host_data.rho = self.rho.cpu().numpy()
        host_data.tau = self.tau.cpu().numpy()
        host_data.Fg = self.Fg.cpu().numpy()
        host_data.nodetype = self.nodetype.cpu().numpy()
        host_data.ex = self.ex.cpu().numpy()
        host_data.ey = self.ey.cpu().numpy()
        host_data.w = self.w.cpu().numpy()

        return host_data

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def compute_edf(self):
        u2 = self.u * self.u
        uxy = self.u[0] * self.u[1]
        u2_sum = torch.sum(u2, dim=0)

        inv_es_sq = 1.0 / (self.es * self.es)

        euxy = torch.outer(self.ex * self.ey, uxy.flatten()).reshape(self.f.shape)
        euxx = torch.outer(self.ex * self.ex, u2[0].flatten()).reshape(self.f.shape)
        euyy = torch.outer(self.ey * self.ey, u2[1].flatten()).reshape(self.f.shape)
        eu2 = 2.0 * euxy + euxx + euyy

        term1 = inv_es_sq * (
            torch.outer(self.ex, self.u[0].flatten())
            + torch.outer(self.ey, self.u[1].flatten())
        ).reshape(self.f.shape)
        term2 = 0.5 * inv_es_sq * (inv_es_sq * eu2 - u2_sum).reshape(self.f.shape)
        f_new = torch.outer(self.w, self.rho.flatten()).reshape(self.f.shape) * (
            1.0 + term1 + term2
        )
        s = (self.nodetype <= 0).repeat(9, 1, 1)
        self.f[s] = f_new[s]

    def compute_macro_vars(self):
        s = self.nodetype <= 0
        self.rho = torch.where(
            s, torch.tensordot(self.f, torch.ones((9)).to(self.device), ([0], [0])), 0.0
        )

        self.u = torch.where(
            s,
            torch.stack(
                (
                    torch.tensordot(self.f, self.ex, ([0], [0])),
                    torch.tensordot(self.f, self.ey, ([0], [0])),
                ),
            )
            / self.rho,
            0.0,
        )

    def collide(self):
        tau_per_rho = torch.where(
            self.rho != 0.0,
            self.tau / self.rho,
            torch.finfo(self.f.dtype).max,
        )

        s = self.nodetype <= 0
        s2 = s.repeat(2, 1, 1)
        self.u[s2] += (self.Fg * tau_per_rho)[s2]

        u2 = self.u * self.u
        u2_p_u = u2 + self.u
        u2_m_u = u2 - self.u
        uxy3 = 3.0 * self.u[0] * self.u[1]

        multipliers = torch.tensor(
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
        ).to(self.device)

        f_updated = torch.cat(
            (
                -u2[0] - u2[1] + 0.333333,
                u2_p_u[0] - 0.5 * u2[1],
                u2_p_u[1] - 0.5 * u2[0],
                u2_p_u[0] + u2_p_u[1] + uxy3,
                u2_p_u[0] + u2_m_u[1] - uxy3,
                u2_m_u[0] - 0.5 * u2[1],
                u2_m_u[1] - 0.5 * u2[0],
                u2_m_u[0] + u2_m_u[1] + uxy3,
                u2_m_u[0] + u2_p_u[1] - uxy3,
            )
        ).reshape(self.f.shape)

        rho_per_three = self.rho * 0.3333333333
        inv_tau = torch.where(
            self.tau != 0.0,
            1.0 / self.tau,
            torch.finfo(self.f.dtype).max,
        )

        f_eq = torch.outer(multipliers, rho_per_three.flatten()).reshape(
            self.f.shape
        ) * (f_updated + 0.33333333)

        f_new = (1.0 - inv_tau) * self.f + inv_tau * f_eq
        s = s.repeat(9, 1, 1)
        self.f[s] = f_new[[0, 5, 6, 7, 8, 1, 2, 3, 4]][s]

    def stream_and_bounce(self):
        ny = self.tau.shape[0]
        nx = self.tau.shape[1]
        q, i, j = torch.meshgrid(
            torch.arange(1, 5).to(self.device),
            torch.arange(ny).to(self.device),
            torch.arange(nx).to(self.device),
            indexing="ij",
        )

        i = i.flatten()
        j = j.flatten()
        q = q.flatten()

        nexti = ((ny + i - self.ey[q]) % ny).type(torch.int32)
        nextj = ((nx + j - self.ex[q]) % nx).type(torch.int32)

        s = ((self.nodetype[i, j] <= 0) & (self.nodetype[nexti, nextj] <= 0)).flatten()

        f_copy = torch.empty_like(self.f).copy_(self.f)

        self.f[q, nexti, nextj][s] = f_copy[q + 4, i, j][s]
        self.f[q + 4, i, j][s] = f_copy[q, nexti, nextj][s]

    def finish(self):
        pass


if __name__ == "__main__":
    run(TorchLBM())
