import cupy as cp

def equilibrium_cupy(rho, ux, uy, uz):
    """
    Compute the equilibrium distribution function (EDF) using CuPy (D3Q19).
    Arguments:
        rho : 3D CuPy array : Local fluid density
        ux, uy, uz : 3D CuPy arrays : Local fluid velocity components
    Returns:
        feq : 4D CuPy array : Equilibrium distribution function (D3Q19)
    """

    # Scale velocity components (1 / c_s^2 = 3.0)
    local_ux = ux * 3.0
    local_uy = uy * 3.0
    local_uz = uz * 3.0

    # Compute common terms
    c3 = -3.0 * (local_ux**2 + local_uy**2 + local_uz**2)
    rhom1 = rho - 1.0  # Shifted term (rho - 1)

    # Rest direction (000)
    feq = cp.zeros((*rho.shape, 19), dtype=cp.float32)
    feq[..., 0] = (1 / 3) * (rho + 0.5 * c3 + rhom1)

    # Precompute velocity terms
    u0 = local_ux + local_uy
    u1 = local_ux + local_uz
    u2 = local_uy + local_uz
    u3 = local_ux - local_uy
    u4 = local_ux - local_uz
    u5 = local_uy - local_uz

    # Weights for cardinal and diagonal directions
    rhos = (1 / 18) * rho  # Cardinal direction weight
    rhoe = (1 / 36) * rho  # Diagonal direction weight
    rhom1s = (1 / 18) * rhom1
    rhom1e = (1 / 36) * rhom1

    # Cardinal directions (+00, -00, 0+0, 0-0, 00+, 00-)
    feq[..., 1] = rhos * (0.5 * (local_ux**2 + c3) + local_ux) + rhom1s
    feq[..., 2] = rhos * (0.5 * (local_ux**2 + c3) - local_ux) + rhom1s
    feq[..., 3] = rhos * (0.5 * (local_uy**2 + c3) + local_uy) + rhom1s
    feq[..., 4] = rhos * (0.5 * (local_uy**2 + c3) - local_uy) + rhom1s
    feq[..., 5] = rhos * (0.5 * (local_uz**2 + c3) + local_uz) + rhom1s
    feq[..., 6] = rhos * (0.5 * (local_uz**2 + c3) - local_uz) + rhom1s

    # Diagonal directions (++0, --0, +0+, -0-, 0++, 0--, +-0, -+0, +0-, -0+, 0+-, 0-+)
    feq[..., 7] = rhoe * (0.5 * (u0**2 + c3) + u0) + rhom1e
    feq[..., 8] = rhoe * (0.5 * (u0**2 + c3) - u0) + rhom1e
    feq[..., 9] = rhoe * (0.5 * (u1**2 + c3) + u1) + rhom1e
    feq[..., 10] = rhoe * (0.5 * (u1**2 + c3) - u1) + rhom1e
    feq[..., 11] = rhoe * (0.5 * (u2**2 + c3) + u2) + rhom1e
    feq[..., 12] = rhoe * (0.5 * (u2**2 + c3) - u2) + rhom1e
    feq[..., 13] = rhoe * (0.5 * (u3**2 + c3) + u3) + rhom1e
    feq[..., 14] = rhoe * (0.5 * (u3**2 + c3) - u3) + rhom1e
    feq[..., 15] = rhoe * (0.5 * (u4**2 + c3) + u4) + rhom1e
    feq[..., 16] = rhoe * (0.5 * (u4**2 + c3) - u4) + rhom1e
    feq[..., 17] = rhoe * (0.5 * (u5**2 + c3) + u5) + rhom1e
    feq[..., 18] = rhoe * (0.5 * (u5**2 + c3) - u5) + rhom1e

    return feq

# Example usage: Initialize arrays on GPU
domain_size = 100  # Example domain size (100x100x100)
rho = cp.ones((domain_size, domain_size, domain_size), dtype=cp.float32)
ux = cp.zeros((domain_size, domain_size, domain_size), dtype=cp.float32)
uy = cp.zeros((domain_size, domain_size, domain_size), dtype=cp.float32)
uz = cp.zeros((domain_size, domain_size, domain_size), dtype=cp.float32)

# Compute equilibrium distribution function on GPU
feq = equilibrium_cupy(rho, ux, uy, uz)

print("EDF for the entire 3D domain calculated using CuPy.")
