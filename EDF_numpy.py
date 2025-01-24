import numpy as np

def sq(x):
    """Square of a value."""
    return x * x

def equilibrium(rho, ux, uy, uz):
    """
    Calculate equilibrium distribution function (EDF) for the D3Q19 model.
    Arguments:
        rho : float : Local fluid density
        ux, uy, uz : float : Local fluid velocity components
    Returns:
        feq : numpy array : Equilibrium distribution values for all 19 directions
    """
    feq = np.zeros(19, dtype=np.float32)

    # Precompute terms related to the velocity and density
    c3 = -3.0 * (sq(ux) + sq(uy) + sq(uz))  # -u^2 / (2 * c_s^2), where c_s^2 = 1/3
    rhom1 = rho - 1.0  # Shifted term (rho - 1)

    # Scale velocity components by 3 (corresponds to 1 / c_s^2 where c_s^2 = 1/3)
    ux *= 3.0
    uy *= 3.0
    uz *= 3.0

    # Rest direction (000)
    feq[0] = (1/3) * (rho + 0.5 * c3 + rhom1)  # w0 * (rho + 0.5 * (-u^2) + (rho - 1))

    # Precompute combined velocity terms
    u0 = ux + uy
    u1 = ux + uz
    u2 = uy + uz
    u3 = ux - uy
    u4 = ux - uz
    u5 = uy - uz

    # Weights for cardinal and diagonal directions
    rhos = (1/18) * rho  # Weight for cardinal directions (ws * rho)
    rhoe = (1/36) * rho  # Weight for diagonal directions (we * rho)
    rhom1s = (1/18) * rhom1  # Shifted term for cardinal directions (ws * (rho - 1))
    rhom1e = (1/36) * rhom1  # Shifted term for diagonal directions (we * (rho - 1))

    # Cardinal directions (+00, -00, 0+0, 0-0, 00+, 00-)
    feq[1] = rhos * (0.5 * (sq(ux) + c3) + ux) + rhom1s
    feq[2] = rhos * (0.5 * (sq(ux) + c3) - ux) + rhom1s
    feq[3] = rhos * (0.5 * (sq(uy) + c3) + uy) + rhom1s
    feq[4] = rhos * (0.5 * (sq(uy) + c3) - uy) + rhom1s
    feq[5] = rhos * (0.5 * (sq(uz) + c3) + uz) + rhom1s
    feq[6] = rhos * (0.5 * (sq(uz) + c3) - uz) + rhom1s

    # Diagonal directions (++0, --0, +0+, -0-, 0++, 0--, +-0, -+0, +0-, -0+, 0+-, 0-+)
    feq[7] = rhoe * (0.5 * (sq(u0) + c3) + u0) + rhom1e
    feq[8] = rhoe * (0.5 * (sq(u0) + c3) - u0) + rhom1e
    feq[9] = rhoe * (0.5 * (sq(u1) + c3) + u1) + rhom1e
    feq[10] = rhoe * (0.5 * (sq(u1) + c3) - u1) + rhom1e
    feq[11] = rhoe * (0.5 * (sq(u2) + c3) + u2) + rhom1e
    feq[12] = rhoe * (0.5 * (sq(u2) + c3) - u2) + rhom1e
    feq[13] = rhoe * (0.5 * (sq(u3) + c3) + u3) + rhom1e
    feq[14] = rhoe * (0.5 * (sq(u3) + c3) - u3) + rhom1e
    feq[15] = rhoe * (0.5 * (sq(u4) + c3) + u4) + rhom1e
    feq[16] = rhoe * (0.5 * (sq(u4) + c3) - u4) + rhom1e
    feq[17] = rhoe * (0.5 * (sq(u5) + c3) + u5) + rhom1e
    feq[18] = rhoe * (0.5 * (sq(u5) + c3) - u5) + rhom1e

    return feq
