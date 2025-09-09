import sys
import numpy as np
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

        self.profile_pytorch = j["profile_pytorch"]
        self.dll = j["dll"]
        self.datadir = j["datadir"]
        self.dtype = np.float32 if j["dtype"] == "float32" else np.float64
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

    def output(self, inputs, elapsed):
        prefix = inputs.datadir + "/"
        postfix = (
            "_" + inputs.output_filename + "_" + np.dtype(inputs.dtype).name + ".csv"
        )

        mid = self.rho.shape[1] // 2
        profiles = np.stack(
            (
                np.arange(self.rho.shape[1]),
                self.rho[:, mid],
                self.u[0, :, mid],
                self.u[1, :, mid],
                self.f[0, :, mid],
                self.f[1, :, mid],
                self.f[2, :, mid],
                self.f[3, :, mid],
                self.f[4, :, mid],
                self.f[5, :, mid],
                self.f[6, :, mid],
                self.f[7, :, mid],
                self.f[8, :, mid],
            )
        )

        np.savetxt(prefix + "profiles" + postfix, profiles.transpose(), delimiter=",")

        runtimes_fname = (
            prefix
            + inputs.output_filename
            + "_"
            + str(inputs.nx)
            + "_"
            + str(inputs.ny)
            + "_"
            + str(inputs.niters)
            + "_"
            + np.dtype(inputs.dtype).name
            + ".txt"
        )

        # Append runtime to a file
        with open(runtimes_fname, "a") as f:
            f.write(str(elapsed) + "\n")


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

    host_data = lbm_impl.copy_to_host(host_data)
    host_data.output(inputs, t1 - t0)

    lbm_impl.finish()
