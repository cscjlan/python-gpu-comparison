import cupy as cp
import matplotlib.pyplot as plt
import time

dtype = cp.float32

# LBM constants
ex = cp.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey = cp.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
es = cp.sqrt(1 / 3)
w0, ws, wl = 4 / 9, 1 / 9, 1 / 36
w = cp.array([w0, ws, ws, wl, wl, ws, ws, wl, wl], dtype=dtype)

def compute_macro_vars(f, nodetype, rho, u):
    """
    Compute macroscopic variables: density (rho) and velocity (u).
    Uses vectorized summation in CuPy for high performance on GPU.
    """

    # Compute density (rho): sum over all velocity directions
    rho[:] = cp.sum(f, axis=2)  # Shape: (ny, nx)

    # Compute velocity components in a single step (vectorized sum + division)
    u[0] = (cp.sum(f[:, :, [1, 3, 8]], axis=2) - cp.sum(f[:, :, [5, 7, 4]], axis=2)) / rho
    u[1] = (cp.sum(f[:, :, [2, 3, 4]], axis=2) - cp.sum(f[:, :, [6, 7, 8]], axis=2)) / rho

    # Apply mask: Set velocity to zero where nodetype > 0 (solid nodes)
    u[:, nodetype > 0] = 0


def compute_edf(rho, u, nodetype):
    """
    Compute the equilibrium distribution function (feq) for LBM.
    Uses precomputed constants for maximum GPU efficiency.
    """

    # Precompute constants to avoid redundant calculations
    inv_es2 = 1.0 / es**2
    inv_es4 = 0.5 / es**4
    inv_3 = 1.0 / 3.0
    inv_6 = 1.0 / 6.0
    inv_9 = 1.0 / 9.0
    inv_12 = 1.0 / 12.0
    inv_24 = 1.0 / 24.0
    inv_36 = 1.0 / 36.0
    inv_8 = 1.0 / 8.0
    inv_2_3 = -2.0 / 3.0  # -2/3
    inv_1_3 = 1.0 / 3.0   # 1/3

    # Precompute squared velocity terms
    u2 = u[0]**2 + u[1]**2

    # Compute feq using precomputed constants
    feq = cp.zeros((9, *rho.shape), dtype=rho.dtype)
    
    feq[0] = rho * (inv_2_3 * u[0]**2 + inv_2_3 * u[1]**2 + 4.0 * inv_9)
    feq[1] = rho * (inv_1_3 * u[0]**2 + inv_1_3 * u[0] - inv_6 * u[1]**2 + inv_9)
    feq[2] = rho * (-inv_6 * u[0]**2 + inv_1_3 * u[1]**2 + inv_1_3 * u[1] + inv_9)
    feq[3] = rho * (-inv_24 * u[0]**2 + inv_12 * u[0] - inv_24 * u[1]**2 + inv_12 * u[1] + inv_8 * (u[0] + u[1])**2 + inv_36)
    feq[4] = rho * (-inv_24 * u[0]**2 + inv_12 * u[0] - inv_24 * u[1]**2 - inv_12 * u[1] + inv_8 * (u[0] - u[1])**2 + inv_36)
    feq[5] = rho * (inv_1_3 * u[0]**2 - inv_1_3 * u[0] - inv_6 * u[1]**2 + inv_9)
    feq[6] = rho * (-inv_6 * u[0]**2 + inv_1_3 * u[1]**2 - inv_1_3 * u[1] + inv_9)
    feq[7] = rho * (-inv_24 * u[0]**2 - inv_12 * u[0] - inv_24 * u[1]**2 - inv_12 * u[1] + inv_8 * (-u[0] - u[1])**2 + inv_36)
    feq[8] = rho * (-inv_24 * u[0]**2 - inv_12 * u[0] - inv_24 * u[1]**2 + inv_12 * u[1] + inv_8 * (-u[0] + u[1])**2 + inv_36)

    # Apply mask: Set feq to zero where nodetype > 0 (solid nodes)
    feq[:, nodetype > 0] = 0

    return feq
#########################

def collide(f, rho, u, nodetype, tau, Fg):
    """
    Perform collision step in Lattice Boltzmann Method (LBM).
    Computes equilibrium distribution function (EDF) directly inside.
    Uses vectorized force application and condition checks for efficiency.
    """

    # ==============================
    # 🚀 Apply External Force (Fully Vectorized Without Extra Mask)
    # ==============================
    u[:, nodetype <= 0] += (Fg[:, nodetype <= 0] * tau[nodetype <= 0]) / rho[nodetype <= 0]  # ✅ Faster Boolean Indexing

    # ==============================
    # 🚀 Precompute Constants
    # ==============================
    inv_es2 = 1.0 / es**2
    inv_es4 = 0.5 / es**4
    inv_3 = 1.0 / 3.0
    inv_6 = 1.0 / 6.0
    inv_9 = 1.0 / 9.0
    inv_12 = 1.0 / 12.0
    inv_24 = 1.0 / 24.0
    inv_36 = 1.0 / 36.0
    inv_8 = 1.0 / 8.0
    inv_2_3 = -2.0 / 3.0
    inv_1_3 = 1.0 / 3.0

    # Precompute squared velocity terms
    u2 = u[0]**2 + u[1]**2

    # ==============================
    # 🚀 Compute Equilibrium Distribution Function (feq)
    # ==============================
    feq = cp.zeros((9, *rho.shape), dtype=rho.dtype)

    feq[0] = rho * (inv_2_3 * u[0]**2 + inv_2_3 * u[1]**2 + 4.0 * inv_9)
    feq[1] = rho * (inv_1_3 * u[0]**2 + inv_1_3 * u[0] - inv_6 * u[1]**2 + inv_9)
    feq[2] = rho * (-inv_6 * u[0]**2 + inv_1_3 * u[1]**2 + inv_1_3 * u[1] + inv_9)
    feq[3] = rho * (-inv_24 * u[0]**2 + inv_12 * u[0] - inv_24 * u[1]**2 + inv_12 * u[1] + inv_8 * (u[0] + u[1])**2 + inv_36)
    feq[4] = rho * (-inv_24 * u[0]**2 + inv_12 * u[0] - inv_24 * u[1]**2 - inv_12 * u[1] + inv_8 * (u[0] - u[1])**2 + inv_36)
    feq[5] = rho * (inv_1_3 * u[0]**2 - inv_1_3 * u[0] - inv_6 * u[1]**2 + inv_9)
    feq[6] = rho * (-inv_6 * u[0]**2 + inv_1_3 * u[1]**2 - inv_1_3 * u[1] + inv_9)
    feq[7] = rho * (-inv_24 * u[0]**2 - inv_12 * u[0] - inv_24 * u[1]**2 - inv_12 * u[1] + inv_8 * (-u[0] - u[1])**2 + inv_36)
    feq[8] = rho * (-inv_24 * u[0]**2 - inv_12 * u[0] - inv_24 * u[1]**2 + inv_12 * u[1] + inv_8 * (-u[0] + u[1])**2 + inv_36)

    # ==============================
    # 🚀 Apply Collision Step (Fully Vectorized Without Mask Variable)
    # ==============================
    f[nodetype <= 0] = (1.0 - (1.0 / tau[nodetype <= 0, None])) * f[nodetype <= 0] + (1.0 / tau[nodetype <= 0, None]) * feq[nodetype <= 0]  # ✅ Faster Boolean Indexing
