import torch


def trace_handler(p):
    p.export_chrome_trace("traces/trace_" + str(p.step_num) + ".json")


class TorchLBM:
    def initialize(self, host_data, inputs):
        self.device = torch.device("cuda")

        self.prof = None
        if inputs.profile_pytorch:
            from torch.profiler import profile, ProfilerActivity

            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                schedule=torch.profiler.schedule(
                    skip_first=10, wait=5, warmup=1, active=3, repeat=2
                ),
                on_trace_ready=trace_handler,
            )

        self.f = torch.from_numpy(host_data.f).to(self.device)
        self.f_updated = torch.from_numpy(host_data.f).to(self.device)
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
        torch.utils.swap_tensors(self.f, self.f_updated)
        self.stream_and_bounce()
        self.compute_macro_vars()
        if self.prof:
            self.prof.step()

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

    @torch.profiler.record_function("compute_edf")
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
        self.f = torch.outer(self.w, self.rho.flatten()).reshape(self.f.shape) * (
            1.0 + term1 + term2
        )

        s = (self.nodetype <= 0).type(self.f.dtype)
        r = 1.0 - s
        self.u += s * self.Fg * self.tau / (self.rho + r)

    @torch.profiler.record_function("compute_macro_vars")
    @torch.no_grad()
    def compute_macro_vars(self):
        s1 = (self.nodetype <= 0).type(self.f.dtype)
        s2 = 1.0 - s1
        self.rho = s1 * torch.sum(self.f, dim=0)

        self.u[0] = (
            s1
            * (self.f[1] + self.f[3] + self.f[4] - self.f[5] - self.f[7] - self.f[8])
            / (self.rho + s2)
        )

        self.u[1] = (
            s1
            * (self.f[2] + self.f[3] - self.f[4] - self.f[6] - self.f[7] + self.f[8])
            / (self.rho + s2)
        )

        self.u += s1 * self.Fg * self.tau / (self.rho + s2)

    @torch.profiler.record_function("collide")
    @torch.no_grad()
    def collide(self):
        s1 = (self.nodetype <= 0).type(self.f.dtype)
        s2 = 1.0 - s1

        u2 = self.u * self.u
        u2_p_u = u2 + self.u
        u2_m_u = u2 - self.u
        uxy3 = 3.0 * self.u[0] * self.u[1]

        inv_tau = 1.0 / self.tau
        one_m_inv_tau = 1.0 - inv_tau
        third_rho_per_tau = 0.3333333333 * self.rho * inv_tau

        f_new = lambda old, mul, feq: (
            one_m_inv_tau * old + third_rho_per_tau * mul * (feq + 0.333333)
        )

        def update_pair(q, feqq, feql, mul):
            l = q + 4
            fq = self.f[q]
            fl = self.f[l]
            self.f_updated[q] = s1 * f_new(fl, mul, feql) + s2 * fq
            self.f_updated[l] = s1 * f_new(fq, mul, feqq) + s2 * fl

        self.f_updated[0] = (
            s1 * f_new(self.f[0], 2.00, -u2[0] - u2[1] + 0.333333) + s2 * self.f[0]
        )

        update_pair(1, u2_p_u[0] - 0.5 * u2[1], u2_m_u[0] - 0.5 * u2[1], 1.00)
        update_pair(2, u2_p_u[1] - 0.5 * u2[0], u2_m_u[1] - 0.5 * u2[0], 1.00)
        update_pair(3, u2_p_u[0] + u2_p_u[1] + uxy3, u2_m_u[0] + u2_m_u[1] + uxy3, 0.25)
        update_pair(4, u2_p_u[0] + u2_m_u[1] - uxy3, u2_m_u[0] + u2_p_u[1] - uxy3, 0.25)

    @torch.profiler.record_function("stream_and_bounce")
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

        s1 = ((self.nodetype[i, j] <= 0) & (self.nodetype[nexti, nextj] <= 0)).type(
            self.f.dtype
        )
        s2 = 1.0 - s1

        f1 = self.f[q, nexti, nextj]
        f2 = self.f[q + 4, i, j]
        self.f[q, nexti, nextj] = s1 * f2 + s2 * f1
        self.f[q + 4, i, j] = s1 * f1 + s2 * f2

    @torch.profiler.record_function("collide_stream_and_bounce")
    @torch.no_grad()
    def collide_stream_and_bounce(self):
        s = (self.nodetype <= 0).type(self.f.dtype)

        u2 = self.u * self.u
        u2_p_u = u2 + self.u
        u2_m_u = u2 - self.u
        uxy3 = 3.0 * self.u[0] * self.u[1]

        inv_tau = 1.0 / self.tau
        one_m_inv_tau = 1.0 - inv_tau
        third_rho_per_tau = 0.3333333333 * self.rho * inv_tau

        f_new = lambda old, mul, feq: (
            one_m_inv_tau * old + third_rho_per_tau * mul * (feq + 0.333333)
        )

        ny = self.tau.shape[0]
        nx = self.tau.shape[1]
        i = torch.arange(ny).repeat_interleave(nx)
        j = torch.arange(nx).repeat(ny)

        # TODO still wrong
        def stream(q, mul, feq):
            nexti = ((ny + i - self.ey[q]) % ny).type(torch.int32)
            nextj = ((nx + j + self.ex[q]) % nx).type(torch.int32)

            s2 = s * (self.nodetype[nexti, nextj] <= 0).type(self.f.dtype).view(s.shape)
            r2 = 1.0 - s2
            fq = self.f[q][nexti, nextj]

            return s2 * f_new(fq, mul, feq) + r2 * self.f[q]

        self.f_updated[0] = stream(0, 2.00, -u2[0] - u2[1] + 0.333333)
        self.f_updated[1] = stream(1, 1.00, u2_p_u[0] - 0.5 * u2[1])
        self.f_updated[2] = stream(2, 1.00, u2_p_u[1] - 0.5 * u2[0])
        self.f_updated[3] = stream(3, 0.25, u2_p_u[0] + u2_p_u[1] + uxy3)
        self.f_updated[4] = stream(4, 0.25, u2_p_u[0] + u2_m_u[1] - uxy3)
        self.f_updated[5] = stream(5, 1.00, u2_m_u[0] - 0.5 * u2[1])
        self.f_updated[6] = stream(6, 1.00, u2_m_u[1] - 0.5 * u2[0])
        self.f_updated[7] = stream(7, 0.25, u2_m_u[0] + u2_m_u[1] + uxy3)
        self.f_updated[8] = stream(8, 0.25, u2_m_u[0] + u2_p_u[1] - uxy3)


if __name__ == "__main__":
    from boilerplate.runner import run

    run(TorchLBM())
