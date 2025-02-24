import cupy as cp

def equilibrium_gpu_cupy(rho_d, ux_d, uy_d, uz_d, feq_d, grid, block):
    """
    Compute equilibrium distribution function (EDF) using CuPy (D3Q19) on GPU.
    Arguments:
        rho_d, ux_d, uy_d, uz_d : 3D CuPy device arrays (input)
        feq_d : 4D CuPy device array (output)
        grid, block : GPU launch configuration
    """
    x, y, z = cp.meshgrid(
        cp.arange(rho_d.shape[0]), cp.arange(rho_d.shape[1]), cp.arange(rho_d.shape[2]), indexing='ij'
    )

    # Local velocity and density
    local_rho = rho_d[x, y, z]
    local_ux = ux_d[x, y, z] * 3.0
    local_uy = uy_d[x, y, z] * 3.0
    local_uz = uz_d[x, y, z] * 3.0

    c3 = -3.0 * (local_ux**2 + local_uy**2 + local_uz**2)
    rhom1 = local_rho - 1.0

    # Rest direction (000)
    feq_d[x, y, z, 0] = (1 / 3) * (local_rho + 0.5 * c3 + rhom1)

    # Velocity combinations
    u0 = local_ux + local_uy
    u1 = local_ux + local_uz
    u2 = local_uy + local_uz
    u3 = local_ux - local_uy
    u4 = local_ux - local_uz
    u5 = local_uy - local_uz

    # Weights
    rhos = (1 / 18) * local_rho
    rhoe = (1 / 36) * local_rho
    rhom1s = (1 / 18) * rhom1
    rhom1e = (1 / 36) * rhom1

    # Cardinal directions
    feq_d[x, y, z, 1] = rhos * (0.5 * (local_ux**2 + c3) + local_ux) + rhom1s
    feq_d[x, y, z, 2] = rhos * (0.5 * (local_ux**2 + c3) - local_ux) + rhom1s
    feq_d[x, y, z, 3] = rhos * (0.5 * (local_uy**2 + c3) + local_uy) + rhom1s
    feq_d[x, y, z, 4] = rhos * (0.5 * (local_uy**2 + c3) - local_uy) + rhom1s
    feq_d[x, y, z, 5] = rhos * (0.5 * (local_uz**2 + c3) + local_uz) + rhom1s
    feq_d[x, y, z, 6] = rhos * (0.5 * (local_uz**2 + c3) - local_uz) + rhom1s

    # Diagonal directions
    feq_d[x, y, z, 7] = rhoe * (0.5 * (u0**2 + c3) + u0) + rhom1e
    feq_d[x, y, z, 8] = rhoe * (0.5 * (u0**2 + c3) - u0) + rhom1e
    feq_d[x, y, z, 9] = rhoe * (0.5 * (u1**2 + c3) + u1) + rhom1e
    feq_d[x, y, z, 10] = rhoe * (0.5 * (u1**2 + c3) - u1) + rhom1e
    feq_d[x, y, z, 11] = rhoe * (0.5 * (u2**2 + c3) + u2) + rhom1e
    feq_d[x, y, z, 12] = rhoe * (0.5 * (u2**2 + c3) - u2) + rhom1e
    feq_d[x, y, z, 13] = rhoe * (0.5 * (u3**2 + c3) + u3) + rhom1e
    feq_d[x, y, z, 14] = rhoe * (0.5 * (u3**2 + c3) - u3) + rhom1e
    feq_d[x, y, z, 15] = rhoe * (0.5 * (u4**2 + c3) + u4) + rhom1e
    feq_d[x, y, z, 16] = rhoe * (0.5 * (u4**2 + c3) - u4) + rhom1e
    feq_d[x, y, z, 17] = rhoe * (0.5 * (u5**2 + c3) + u5) + rhom1e
    feq_d[x, y, z, 18] = rhoe * (0.5 * (u5**2 + c3) - u5) + rhom1e

# --- Example Usage ---
domain_size = 100

# Allocate device arrays
rho_d = cp.ones((domain_size, domain_size, domain_size), dtype=cp.float32)
ux_d = cp.zeros((domain_size, domain_size, domain_size), dtype=cp.float32)
uy_d = cp.zeros((domain_size, domain_size, domain_size), dtype=cp.float32)
uz_d = cp.zeros((domain_size, domain_size, domain_size), dtype=cp.float32)
feq_d = cp.zeros((domain_size, domain_size, domain_size, 19), dtype=cp.float32)

# Launch kernel with the same grid/block config as Numba
threads_per_block = (8, 8, 8)
blocks_per_grid = (
    (domain_size + threads_per_block[0] - 1) // threads_per_block[0],
    (domain_size + threads_per_block[1] - 1) // threads_per_block[1],
    (domain_size + threads_per_block[2] - 1) // threads_per_block[2]
)

equilibrium_gpu_cupy(rho_d, ux_d, uy_d, uz_d, feq_d, blocks_per_grid, threads_per_block)

# Copy results back to host
feq = feq_d.get()

print("EDF computed using CuPy.")
