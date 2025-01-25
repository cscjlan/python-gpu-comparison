import numpy as np
from numba import cuda

@cuda.jit
def equilibrium_gpu(rho, ux, uy, uz, feq):
    """
    Calculate equilibrium distribution function (EDF) for the D3Q19 model on GPU.
    Arguments:
        rho : 3D array : Local fluid density
        ux, uy, uz : 3D arrays : Local fluid velocity components
        feq : 4D array : Output array for equilibrium distribution values
    """
    x, y, z = cuda.grid(3)  # Get thread indices for 3D grid

    # Check bounds
    if x < rho.shape[0] and y < rho.shape[1] and z < rho.shape[2]:
        # Local velocity and density values for this thread
        local_rho = rho[x, y, z]
        local_ux = ux[x, y, z] * 3.0  # Scale by 3 (1 / c_s^2)
        local_uy = uy[x, y, z] * 3.0
        local_uz = uz[x, y, z] * 3.0

        # Precompute terms related to the velocity and density
        c3 = -3.0 * (local_ux * local_ux + local_uy * local_uy + local_uz * local_uz)
        rhom1 = local_rho - 1.0  # Shifted term (rho - 1)

        # Rest direction (000)
        feq[x, y, z, 0] = (1 / 3) * (local_rho + 0.5 * c3 + rhom1)

        # Precompute combined velocity terms
        u0 = local_ux + local_uy
        u1 = local_ux + local_uz
        u2 = local_uy + local_uz
        u3 = local_ux - local_uy
        u4 = local_ux - local_uz
        u5 = local_uy - local_uz

        # Weights for cardinal and diagonal directions
        rhos = (1 / 18) * local_rho  # Weight for cardinal directions (ws * rho)
        rhoe = (1 / 36) * local_rho  # Weight for diagonal directions (we * rho)
        rhom1s = (1 / 18) * rhom1  # Shifted term for cardinal directions (ws * (rho - 1))
        rhom1e = (1 / 36) * rhom1  # Shifted term for diagonal directions (we * (rho - 1))

        # Cardinal directions (+00, -00, 0+0, 0-0, 00+, 00-)
        feq[x, y, z, 1] = rhos * (0.5 * (local_ux * local_ux + c3) + local_ux) + rhom1s
        feq[x, y, z, 2] = rhos * (0.5 * (local_ux * local_ux + c3) - local_ux) + rhom1s
        feq[x, y, z, 3] = rhos * (0.5 * (local_uy * local_uy + c3) + local_uy) + rhom1s
        feq[x, y, z, 4] = rhos * (0.5 * (local_uy * local_uy + c3) - local_uy) + rhom1s
        feq[x, y, z, 5] = rhos * (0.5 * (local_uz * local_uz + c3) + local_uz) + rhom1s
        feq[x, y, z, 6] = rhos * (0.5 * (local_uz * local_uz + c3) - local_uz) + rhom1s

        # Diagonal directions (++0, --0, +0+, -0-, 0++, 0--, +-0, -+0, +0-, -0+, 0+-, 0-+)
        feq[x, y, z, 7] = rhoe * (0.5 * (u0 * u0 + c3) + u0) + rhom1e
        feq[x, y, z, 8] = rhoe * (0.5 * (u0 * u0 + c3) - u0) + rhom1e
        feq[x, y, z, 9] = rhoe * (0.5 * (u1 * u1 + c3) + u1) + rhom1e
        feq[x, y, z, 10] = rhoe * (0.5 * (u1 * u1 + c3) - u1) + rhom1e
        feq[x, y, z, 11] = rhoe * (0.5 * (u2 * u2 + c3) + u2) + rhom1e
        feq[x, y, z, 12] = rhoe * (0.5 * (u2 * u2 + c3) - u2) + rhom1e
        feq[x, y, z, 13] = rhoe * (0.5 * (u3 * u3 + c3) + u3) + rhom1e
        feq[x, y, z, 14] = rhoe * (0.5 * (u3 * u3 + c3) - u3) + rhom1e
        feq[x, y, z, 15] = rhoe * (0.5 * (u4 * u4 + c3) + u4) + rhom1e
        feq[x, y, z, 16] = rhoe * (0.5 * (u4 * u4 + c3) - u4) + rhom1e
        feq[x, y, z, 17] = rhoe * (0.5 * (u5 * u5 + c3) + u5) + rhom1e
        feq[x, y, z, 18] = rhoe * (0.5 * (u5 * u5 + c3) - u5) + rhom1e

# Example usage: Initialize arrays and launch kernel
domain_size = 100  # Example domain size (100x100x100)
rho = np.ones((domain_size, domain_size, domain_size), dtype=np.float32)
ux = np.zeros((domain_size, domain_size, domain_size), dtype=np.float32)
uy = np.zeros((domain_size, domain_size, domain_size), dtype=np.float32)
uz = np.zeros((domain_size, domain_size, domain_size), dtype=np.float32)
feq = np.zeros((domain_size, domain_size, domain_size, 19), dtype=np.float32)

# Copy data to device memory
rho_d = cuda.to_device(rho)
ux_d = cuda.to_device(ux)
uy_d = cuda.to_device(uy)
uz_d = cuda.to_device(uz)
feq_d = cuda.to_device(feq)

# Configure the kernel
threads_per_block = (8, 8, 8)
blocks_per_grid = (
    (domain_size + threads_per_block[0] - 1) // threads_per_block[0],
    (domain_size + threads_per_block[1] - 1) // threads_per_block[1],
    (domain_size + threads_per_block[2] - 1) // threads_per_block[2]
)

# Launch the kernel
equilibrium_gpu[blocks_per_grid, threads_per_block](rho_d, ux_d, uy_d, uz_d, feq_d)

# Copy results back to host
feq = feq_d.copy_to_host()
print("EDF for the entire 3D domain calculated on GPU.")
