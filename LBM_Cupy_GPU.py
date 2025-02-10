import cupy as cp
import matplotlib.pyplot as plt
import time

dtype = cp.float64  # Ensure data type is compatible with CuPy

# Define the lattice velocities and weights
ex = cp.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey = cp.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
es = (1/3)**0.5
w0 = 4/9
ws = 1/9
wl = 1/36
w = cp.array([w0, ws, ws, wl, wl, ws, ws, wl, wl], dtype=dtype)



def compute_macro_vars(f, nodetype, rho, u):
    """
    Compute macroscopic variables (density and velocity) for Lattice Boltzmann Method (LBM).
    Uses GPU acceleration via CuPy.

    Parameters:
    - f: CuPy 3D array of shape [ny, nx, 9] (distribution function)
    - nodetype: CuPy 2D array of shape [ny, nx] (node type: fluid or solid)
    - rho: CuPy 2D array of shape [ny, nx] (fluid density)
    - u: CuPy 3D array of shape [2, ny, nx] (velocity components u_x, u_y)
    """

    # Compute density (sum of all f values along the last axis)
    rho[:] = cp.sum(f, axis=2) + 1.0

    # Compute velocity components (only for fluid nodes where nodetype <= 0)
    fdotex = f[:, :, 1] + f[:, :, 3] + f[:, :, 4] - f[:, :, 5] - f[:, :, 7] - f[:, :, 8]
    fdotey = f[:, :, 2] + f[:, :, 3] - f[:, :, 4] - f[:, :, 6] - f[:, :, 7] + f[:, :, 8]

    u[0, :, :] = cp.where(nodetype <= 0, fdotex / rho, 0)  # u_x
    u[1, :, :] = cp.where(nodetype <= 0, fdotey / rho, 0)  # u_y

    # Set rho = 0 where nodetype is not fluid
    rho[nodetype > 0] = 0




