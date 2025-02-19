import cupy as cp

dtype = cp.float64

ex = cp.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey = cp.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
es = cp.sqrt(1 / 3)
w0, ws, wl = 4 / 9, 1 / 9, 1 / 36
w = cp.array([w0, ws, ws, wl, wl, ws, ws, wl, wl], dtype=dtype)


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
    feq = cp.zeros((*rho.shape, 9), dtype=dtype)

    feq[...,0] = rho * (inv_2_3 * u[0]**2 + inv_2_3 * u[1]**2 + 4.0 * inv_9)
    feq[...,1] = rho * (inv_1_3 * u[0]**2 + inv_1_3 * u[0] - inv_6 * u[1]**2 + inv_9)
    feq[...,2] = rho * (-inv_6 * u[0]**2 + inv_1_3 * u[1]**2 + inv_1_3 * u[1] + inv_9)
    feq[...,3] = rho * (-inv_24 * u[0]**2 + inv_12 * u[0] - inv_24 * u[1]**2 + inv_12 * u[1] + inv_8 * (u[0] + u[1])**2 + inv_36)
    feq[...,4] = rho * (-inv_24 * u[0]**2 + inv_12 * u[0] - inv_24 * u[1]**2 - inv_12 * u[1] + inv_8 * (u[0] - u[1])**2 + inv_36)
    feq[...,5] = rho * (inv_1_3 * u[0]**2 - inv_1_3 * u[0] - inv_6 * u[1]**2 + inv_9)
    feq[...,6] = rho * (-inv_6 * u[0]**2 + inv_1_3 * u[1]**2 - inv_1_3 * u[1] + inv_9)
    feq[...,7] = rho * (-inv_24 * u[0]**2 - inv_12 * u[0] - inv_24 * u[1]**2 - inv_12 * u[1] + inv_8 * (-u[0] - u[1])**2 + inv_36)
    feq[...,8] = rho * (-inv_24 * u[0]**2 - inv_12 * u[0] - inv_24 * u[1]**2 + inv_12 * u[1] + inv_8 * (-u[0] + u[1])**2 + inv_36)

    # ==============================
    # 🚀 Apply Collision Step (Fully Vectorized Without Mask Variable)
    # ==============================
    f[nodetype <= 0, :] = (1.0 - (1.0 / tau[nodetype <= 0, None])) * f[nodetype <= 0, :] + (1.0 / tau[nodetype <= 0, None]) * feq[nodetype <= 0, :] # ✅ Faster Boolean Indexing

# ================================
# 🚀 TEST FUNCTION
# ================================
def test_collide():
    """Test function for collide()."""

    # Test grid size
    ny, nx = 10, 10  # Small grid for testing

    # Initialize test arrays
    f = cp.ones((ny, nx, 9), dtype=dtype)  # Initial distribution function
    rho = cp.ones((ny, nx), dtype=dtype)  # Uniform density field
    u = cp.zeros((2, ny, nx), dtype=dtype)  # Zero velocity field (fluid at rest)
    tau = cp.ones((ny, nx), dtype=dtype) * 0.8  # Relaxation time
    Fg = cp.zeros((2, ny, nx), dtype=dtype)  # No external force initially
    nodetype = cp.zeros((ny, nx), dtype=dtype)  # All nodes are fluid

    # Run function
    collide(f, rho, u, nodetype, tau, Fg)

    # ✅ Check output shape
    assert f.shape == (ny, nx, 9), f"Expected shape ({ny}, {nx}, 9), but got {f.shape}"

    # ✅ Check if f is CuPy array
    assert isinstance(f, cp.ndarray), "f is not a CuPy array"

    # ✅ Check equilibrium distribution consistency
    feq_expected = cp.zeros((ny, nx,9), dtype=dtype)
    f_expected = cp.ones((ny, nx,9), dtype=dtype)
    feq_expected[...,0] = rho * (-2.0 / 3.0 * u[0]**2 - 2.0 / 3.0 * u[1]**2 + 4.0 / 9.0)
    feq_expected[...,1] = rho * ((1.0 / 3.0) * u[0]**2 + (1.0 / 3.0) * u[0] - 1.0 / 6.0 * u[1]**2 + 1.0 / 9.0)

    f_expected[nodetype <= 0, 0] = (1.0 - (1.0 / tau[nodetype <= 0])) * f_expected[nodetype <= 0, 0] + (1.0 / tau[nodetype <= 0]) * feq_expected[nodetype <= 0, 0]
    f_expected[nodetype <= 0, 1] = (1.0 - (1.0 / tau[nodetype <= 0])) * f_expected[nodetype <= 0, 1] + (1.0 / tau[nodetype <= 0]) * feq_expected[nodetype <= 0, 1]
    print("Computed f[..., 0]:\n", f[..., 0])
    print("Expected feq_expected[0]:\n", f_expected[...,0])
    # Only checking first two for efficiency
    assert cp.allclose(f[..., 0], f_expected[...,0], atol=1e-5), "Collision function incorrect for f[0]"
    assert cp.allclose(f[..., 1], f_expected[...,1], atol=1e-5), "Collision function incorrect for f[1]"

    # ✅ Check that solid nodes are not updated
    nodetype[3, 3] = 1  # Make one solid node
    f_before = f.copy()
    collide(f, rho, u, nodetype, tau, Fg)
    assert cp.allclose(f[3, 3,:], f_before[3, 3,:]), "Solid node should remain unchanged"

    print("✅ All tests passed successfully!")

# ================================
# 🚀 Run the Test
# ================================
test_collide()