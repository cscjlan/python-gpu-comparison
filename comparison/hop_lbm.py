import ctypes
import numpy as np
from boilerplate.runner import run


class Dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("z", ctypes.c_int)]


class HopLBM:
    def initialize(self, host_data):
        self.hop = ctypes.CDLL("./lbm.so")

        self.threads_per_block = Dim3(32, 16)
        self.blocks_per_grid = Dim3(32, 64)

        # These are numpy ndarrays reciding in device memory
        # It's unsafe, but simplifies passing to C library
        self.f = self.to_device(host_data.f)
        self.u = self.to_device(host_data.u)
        self.rho = self.to_device(host_data.rho)
        self.tau = self.to_device(host_data.tau)
        self.Fg = self.to_device(host_data.Fg)
        self.nodetype = self.to_device(host_data.nodetype)
        self.ex = self.to_device(host_data.ex)
        self.ey = self.to_device(host_data.ey)
        self.w = self.to_device(host_data.w)
        self.es = host_data.es
        self.nx = self.rho.shape[1]
        self.ny = self.rho.shape[0]

        if host_data.f.dtype.itemsize == 4:
            self.compute_edf = self.hop.LBM_compute_edf_f32
            self.collide = self.hop.LBM_collide_f32
            self.stream_and_bounce = self.hop.LBM_stream_and_bounce_f32
            self.compute_macro_vars = self.hop.LBM_compute_macro_vars_f32
        else:
            self.compute_edf = self.hop.LBM_compute_edf_f64
            self.collide = self.hop.LBM_collide_f64
            self.stream_and_bounce = self.hop.LBM_stream_and_bounce_f64
            self.compute_macro_vars = self.hop.LBM_compute_macro_vars_f64

        self.compute_edf.argtypes = [
            ctypes.POINTER(Dim3),
            ctypes.POINTER(Dim3),
            ctypes.c_int,
            ctypes.c_int,
            self.make_ndpointer(self.f),
            self.make_ndpointer(self.rho),
            self.make_ndpointer(self.u),
            self.make_ndpointer(self.nodetype),
            self.make_ndpointer(self.ex),
            self.make_ndpointer(self.ey),
            self.make_ndpointer(self.w),
            np.ctypeslib.as_ctypes_type(self.ex.dtype),
        ]

        self.collide.argtypes = [
            ctypes.POINTER(Dim3),
            ctypes.POINTER(Dim3),
            ctypes.c_int,
            ctypes.c_int,
            self.make_ndpointer(self.f),
            self.make_ndpointer(self.rho),
            self.make_ndpointer(self.u),
            self.make_ndpointer(self.nodetype),
            self.make_ndpointer(self.tau),
            self.make_ndpointer(self.Fg),
        ]

        self.stream_and_bounce.argtypes = [
            ctypes.POINTER(Dim3),
            ctypes.POINTER(Dim3),
            ctypes.c_int,
            ctypes.c_int,
            self.make_ndpointer(self.f),
            self.make_ndpointer(self.nodetype),
            self.make_ndpointer(self.ex),
            self.make_ndpointer(self.ey),
        ]

        self.compute_macro_vars.argtypes = [
            ctypes.POINTER(Dim3),
            ctypes.POINTER(Dim3),
            ctypes.c_int,
            ctypes.c_int,
            self.make_ndpointer(self.f),
            self.make_ndpointer(self.rho),
            self.make_ndpointer(self.u),
            self.make_ndpointer(self.nodetype),
            self.make_ndpointer(self.ex),
            self.make_ndpointer(self.ey),
        ]

        self.compute_edf(
            ctypes.byref(self.blocks_per_grid),
            ctypes.byref(self.threads_per_block),
            self.nx,
            self.ny,
            self.f,
            self.rho,
            self.u,
            self.nodetype,
            self.ex,
            self.ey,
            self.w,
            self.es,
        )

    def to_device(self, src: np.ndarray) -> np.ndarray:
        ndptr = self.make_ndpointer(src)

        num_values = np.prod(src.shape)
        sizeof = src.dtype.itemsize
        total_bytes = num_values * sizeof

        # First allocate, then memcpy
        self.hop.LBM_malloc.argtypes = [ctypes.c_size_t]
        self.hop.LBM_malloc.restype = ndptr
        dst = self.hop.LBM_malloc(total_bytes)

        self.hop.LBM_memcpy.argtypes = [ndptr, ndptr, ctypes.c_size_t]
        self.hop.LBM_memcpy(dst, src, total_bytes)

        return np.ctypeslib.as_array(
            ctypes.cast(dst, ctypes.POINTER(np.ctypeslib.as_ctypes_type(src.dtype))),
            shape=src.shape,
        )

    def make_ndpointer(self, src: np.ndarray):
        ndptr = np.ctypeslib.ndpointer(
            dtype=src.dtype, ndim=len(src.shape), shape=src.shape, flags=("C", "A", "W")
        )

        return ndptr

    def to_host(self, dst: np.ndarray, src: np.ndarray):
        ndptr = self.make_ndpointer(src)

        num_values = np.prod(src.shape)
        sizeof = src.dtype.itemsize
        total_bytes = num_values * sizeof

        self.hop.LBM_memcpy.argtypes = [ndptr, ndptr, ctypes.c_size_t]
        self.hop.LBM_memcpy(dst, src, total_bytes)

        return dst

    def free(self, src: np.ndarray):
        self.hop.LBM_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.hop.LBM_free(src)

    def iterate(self):
        self.collide(
            ctypes.byref(self.blocks_per_grid),
            ctypes.byref(self.threads_per_block),
            self.nx,
            self.ny,
            self.f,
            self.rho,
            self.u,
            self.nodetype,
            self.tau,
            self.Fg,
        )
        self.stream_and_bounce(
            ctypes.byref(self.blocks_per_grid),
            ctypes.byref(self.threads_per_block),
            self.nx,
            self.ny,
            self.f,
            self.nodetype,
            self.ex,
            self.ey,
        )
        self.compute_macro_vars(
            ctypes.byref(self.blocks_per_grid),
            ctypes.byref(self.threads_per_block),
            self.nx,
            self.ny,
            self.f,
            self.rho,
            self.u,
            self.nodetype,
            self.ex,
            self.ey,
        )

    def copy_to_host(self, host_data):
        host_data.f = self.to_host(host_data.f, self.f)
        host_data.u = self.to_host(host_data.u, self.u)
        host_data.rho = self.to_host(host_data.rho, self.rho)
        host_data.tau = self.to_host(host_data.tau, self.tau)
        host_data.Fg = self.to_host(host_data.Fg, self.Fg)
        host_data.nodetype = self.to_host(host_data.nodetype, self.nodetype)
        host_data.ex = self.to_host(host_data.ex, self.ex)
        host_data.ey = self.to_host(host_data.ey, self.ey)
        host_data.w = self.to_host(host_data.w, self.w)

        return host_data

    def synchronize(self):
        self.hop.LBM_synchronize()

    def finish(self):
        self.free(self.f)
        self.free(self.u)
        self.free(self.rho)
        self.free(self.tau)
        self.free(self.Fg)
        self.free(self.nodetype)
        self.free(self.ex)
        self.free(self.ey)
        self.free(self.w)


if __name__ == "__main__":
    run(HopLBM())
