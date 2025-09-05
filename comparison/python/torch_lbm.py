import torch


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
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def finish(self):
        pass

    @torch.no_grad()
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

    @torch.no_grad()
    def compute_macro_vars(self):
        s = self.nodetype <= 0
        self.rho[:] = torch.where(
            s,
            torch.tensordot(
                self.f, torch.ones((9)).type(self.f.dtype).to(self.device), ([0], [0])
            ),
            0.0,
        )

        self.u[:] = torch.where(
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

    @torch.no_grad()
    def collide(self):
        s = self.nodetype <= 0
        self.u[:] += torch.where(s, self.Fg * self.tau / self.rho, 0.0)

        u2 = self.u * self.u
        u2_p_u = u2 + self.u
        u2_m_u = u2 - self.u
        uxy3 = 3.0 * self.u[0] * self.u[1]

        rho_per_three = self.rho * 0.3333333333
        inv_tau = torch.where(
            self.tau != 0.0,
            1.0 / self.tau,
            torch.finfo(self.f.dtype).max,
        )

        one_m_inv_tau = 1.0 - inv_tau

        # fmt: off
        self.f[0][s] = (one_m_inv_tau * self.f[0] + 2.00 * inv_tau * rho_per_three * (-u2[0] - u2[1] + 0.333333    + 0.333333))[s]
        self.f[1][s] = (one_m_inv_tau * self.f[1] + 1.00 * inv_tau * rho_per_three * (u2_p_u[0] - 0.5 * u2[1]      + 0.333333))[s]
        self.f[2][s] = (one_m_inv_tau * self.f[2] + 1.00 * inv_tau * rho_per_three * (u2_p_u[1] - 0.5 * u2[0]      + 0.333333))[s]
        self.f[3][s] = (one_m_inv_tau * self.f[3] + 0.25 * inv_tau * rho_per_three * (u2_p_u[0] + u2_p_u[1] + uxy3 + 0.333333))[s]
        self.f[4][s] = (one_m_inv_tau * self.f[4] + 0.25 * inv_tau * rho_per_three * (u2_p_u[0] + u2_m_u[1] - uxy3 + 0.333333))[s]
        self.f[5][s] = (one_m_inv_tau * self.f[5] + 1.00 * inv_tau * rho_per_three * (u2_m_u[0] - 0.5 * u2[1]      + 0.333333))[s]
        self.f[6][s] = (one_m_inv_tau * self.f[6] + 1.00 * inv_tau * rho_per_three * (u2_m_u[1] - 0.5 * u2[0]      + 0.333333))[s]
        self.f[7][s] = (one_m_inv_tau * self.f[7] + 0.25 * inv_tau * rho_per_three * (u2_m_u[0] + u2_m_u[1] + uxy3 + 0.333333))[s]
        self.f[8][s] = (one_m_inv_tau * self.f[8] + 0.25 * inv_tau * rho_per_three * (u2_m_u[0] + u2_p_u[1] - uxy3 + 0.333333))[s]
        # fmt: on

        for q in range(1, 5):
            f_copy = self.f[q].detach().clone()
            self.f[q][s] = self.f[q + 4][s]
            self.f[q + 4][s] = f_copy[s]

    @torch.no_grad()
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
        nextj = ((nx + j + self.ex[q]) % nx).type(torch.int32)

        s = (
            ((self.nodetype[i, j] <= 0) & (self.nodetype[nexti, nextj] <= 0))
            .flatten()
            .type(self.f.dtype)
        )

        f1 = self.f[q, nexti, nextj]
        f2 = self.f[q + 4, i, j]
        self.f[q, nexti, nextj] = (1.0 - s) * f1 + s * f2
        self.f[q + 4, i, j] = (1.0 - s) * f2 + s * f1


if __name__ == "__main__":
    from boilerplate.runner import run

    run(TorchLBM())
