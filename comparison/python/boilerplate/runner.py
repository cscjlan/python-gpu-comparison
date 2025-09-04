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

    def output(self, inputs):
        prefix = inputs.datadir + "/"
        postfix = (
            "_" + inputs.output_filename + "_" + np.dtype(inputs.dtype).name + ".csv"
        )

        np.savetxt(prefix + "rho" + postfix, self.rho, delimiter=",")
        np.savetxt(prefix + "u0" + postfix, self.u[0], delimiter=",")
        np.savetxt(prefix + "u1" + postfix, self.u[1], delimiter=",")
        np.savetxt(prefix + "f0" + postfix, self.f[0], delimiter=",")
        np.savetxt(prefix + "f1" + postfix, self.f[1], delimiter=",")
        np.savetxt(prefix + "f2" + postfix, self.f[2], delimiter=",")
        np.savetxt(prefix + "f3" + postfix, self.f[3], delimiter=",")
        np.savetxt(prefix + "f4" + postfix, self.f[4], delimiter=",")
        np.savetxt(prefix + "f5" + postfix, self.f[5], delimiter=",")
        np.savetxt(prefix + "f6" + postfix, self.f[6], delimiter=",")
        np.savetxt(prefix + "f7" + postfix, self.f[7], delimiter=",")
        np.savetxt(prefix + "f8" + postfix, self.f[8], delimiter=",")


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
