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
    Uses fully vectorized operations in CuPy for high GPU efficiency.
    """

    # Precompute squared velocity terms (avoiding redundant calculations)
    u2 = u[0] ** 2 + u[1] ** 2  # Shape: (ny, nx)

    # Compute first-order term for all velocity directions at once
    Termorder1 = (1.0 / es**2) * (ex[:, None, None] * u[0] + ey[:, None, None] * u[1])

    # Compute second-order term (without loops)
    eu2 = 2 * (ex[:, None, None] * ey[:, None, None] * u[0] * u[1]) + \
          (ex[:, None, None]**2 * u[0]**2) + \
          (ey[:, None, None]**2 * u[1]**2)
    
    Termorder2 = (0.5 / es**4) * eu2 - (0.5 / es**2) * u2[None, :, :]

    # Compute feq in one step (fully vectorized)
    feq = w[:, None, None] * rho[None, :, :] * (1 + Termorder1 + Termorder2)

    # Apply mask: Set feq to zero where nodetype > 0 (solid nodes)
    feq[:, nodetype > 0] = 0

    return feq

def collide(f, rho, u, nodetype, tau, Fg):
    mask = nodetype <= 0
    u[0][mask] += (Fg[0][mask] * tau[mask]) / rho[mask]
    u[1][mask] += (Fg[1][mask] * tau[mask]) / rho[mask]
    
    feq = compute_edf(rho, u, nodetype)
    
    f[mask] = (1.0 - (1.0 / tau[mask, None])) * f[mask] + (1.0 / tau[mask, None]) * feq[mask]
    
    # Swap streaming directions
    for q in range(1, 5):
        f[..., q], f[..., q + 4] = f[..., q + 4], f[..., q]

def stream_and_bounce(f, nodetype):
    ny, nx = nodetype.shape
    f_next = cp.zeros_like(f)
    mask = nodetype <= 0  # Boolean mask for fluid nodes

    # Apply streaming
    for q in range(1, 5):
        nexti = (cp.arange(ny)[:, None] - ey[q]).clip(0, ny - 1).astype(cp.int32)
        nextj = (cp.arange(nx)[None, :] + ex[q]).clip(0, nx - 1).astype(cp.int32)

        valid_mask = mask[nexti, nextj]  # Ensure indices are integers
        f_next[valid_mask, q] = f[valid_mask, q + 4]
        f_next[valid_mask, q + 4] = f[valid_mask, q]

    f[:] = f_next


def test_lb():
    nx, ny, niters = 200, 200, 10000
    rho = cp.ones((ny, nx), dtype=dtype)
    tau = cp.ones((ny, nx), dtype=dtype)
    u = cp.zeros((2, ny, nx), dtype=dtype)
    Fg = cp.zeros((2, ny, nx), dtype=dtype)
    Fg[0, :, :] = 1e-4
    nodetype = cp.zeros((ny, nx), dtype=dtype)
    nodetype[0, :] = 1
    nodetype[-1, :] = 1

    f = compute_edf(rho, u, nodetype)

    t0 = time.time()
    for _ in range(niters):
        collide(f, rho, u, nodetype, tau, Fg)
        stream_and_bounce(f, nodetype)
        compute_macro_vars(f, nodetype, rho, u)
    t1 = time.time()

    mlups = (ny * nx * niters * 1e-6) / (t1 - t0)
    print("MLUPS:", mlups)
    print("Time taken:", t1 - t0)

    # Transfer results to CPU for plotting
    u_cpu = cp.asnumpy(u)

    plt.figure(1)
    plt.quiver(u_cpu[0], u_cpu[1])
    plt.figure(2)
    plt.plot(u_cpu[0][:, nx // 2])
    plt.figure(3)
    plt.imshow(u_cpu[0])
    plt.show()

test_lb()



##################

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

    # ==============================
    # 🚀 Swap Streaming Directions
    # ==============================
    for q in range(1, 5):
        f[..., q], f[..., q + 4] = f[..., q + 4], f[..., q]  # ✅ This remains unchanged
