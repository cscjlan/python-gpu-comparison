import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import json


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

        self.dll = j["dll"]
        self.datadir = j["datadir"]
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

        self.nodetype = np.zeros(shape_1, dtype=np.int32)
        self.nodetype[0, :] = 1
        self.nodetype[-1, :] = 1

        self.f = np.zeros(shape_9, dtype=inputs.dtype)
        self.ex = np.array(inputs.ex, dtype=inputs.dtype)
        self.ey = np.array(inputs.ey, dtype=inputs.dtype)
        self.w = np.array(inputs.w, dtype=inputs.dtype)
        self.es = inputs.dtype(inputs.es)

    def output(self, inputs):
        plt.imsave(inputs.datadir + "/u0_" + inputs.output_filename + ".png", self.u[0])
        plt.imsave(inputs.datadir + "/u1_" + inputs.output_filename + ".png", self.u[1])
        plt.imsave(inputs.datadir + "/f0_" + inputs.output_filename + ".png", self.f[0])
        plt.imsave(inputs.datadir + "/f1_" + inputs.output_filename + ".png", self.f[1])
        plt.imsave(inputs.datadir + "/f2_" + inputs.output_filename + ".png", self.f[2])
        plt.imsave(inputs.datadir + "/f3_" + inputs.output_filename + ".png", self.f[3])
        plt.imsave(inputs.datadir + "/f4_" + inputs.output_filename + ".png", self.f[4])
        plt.imsave(inputs.datadir + "/f5_" + inputs.output_filename + ".png", self.f[5])
        plt.imsave(inputs.datadir + "/f6_" + inputs.output_filename + ".png", self.f[6])
        plt.imsave(inputs.datadir + "/f7_" + inputs.output_filename + ".png", self.f[7])
        plt.imsave(inputs.datadir + "/f8_" + inputs.output_filename + ".png", self.f[8])

        plt.figure()
        x = int(self.u.shape[1] / 2)
        plt.plot(self.u[0][:, x])
        plt.plot(self.u[1][:, x])
        plt.savefig(inputs.datadir + "/profile_u_" + inputs.output_filename + ".png")

        plt.figure()
        x = int(self.f.shape[1] / 2)
        plt.plot(self.f[0][:, x])
        plt.plot(self.f[1][:, x])
        plt.plot(self.f[2][:, x])
        plt.plot(self.f[3][:, x])
        plt.plot(self.f[4][:, x])
        plt.plot(self.f[5][:, x])
        plt.plot(self.f[6][:, x])
        plt.plot(self.f[7][:, x])
        plt.plot(self.f[8][:, x])
        plt.savefig(inputs.datadir + "/profile_f_" + inputs.output_filename + ".png")


def run(lbm_impl):
    inputs = Inputs()
    host_data = HostData(inputs)

    # Initialize and run one iteration to clear the pipes
    lbm_impl.initialize(host_data, inputs)
    lbm_impl.iterate()
    lbm_impl.synchronize()

    t0 = time.time()
    for _ in range(inputs.niters - 1):
        lbm_impl.iterate()
    lbm_impl.synchronize()
    t1 = time.time()

    elapsed = t1 - t0
    mlups = (inputs.ny * inputs.nx * inputs.niters * 1e-6) / elapsed
    print("MLUPS:", mlups)
    print("Time taken", elapsed)

    host_data = lbm_impl.copy_to_host(host_data)
    host_data.output(inputs)
    lbm_impl.finish()
