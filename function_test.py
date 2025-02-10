import cupy as cp

# ==========================
# Function: compute_edf
# ==========================
def compute_edf(rho, u, nodetype):
    """
    Compute the equilibrium distribution function (edf) for Lattice Boltzmann Method (LBM).
    Uses GPU acceleration via CuPy.

    Parameters:
    - rho: CuPy 2D array [ny, nx] (fluid density)
    - u: CuPy 3D array [2, ny, nx] (velocity field u_x, u_y)
    - nodetype: CuPy 2D array [ny, nx] (node type: fluid or solid)
    
    Returns:
    - feq: CuPy 3D array [ny, nx, 9] (equilibrium distribution function)
    """
    
    # Define LBM parameters
    es = cp.sqrt(1/3)  # Speed of sound in lattice units
    w = cp.array([4/9, 1/9, 1/9, 1/36, 1/36, 1/9, 1/9, 1/36, 1/36], dtype=cp.float64)  # LBM weights
    ex = cp.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=cp.float64)
    ey = cp.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=cp.float64)

    ny, nx = nodetype.shape
    feq = cp.zeros((ny, nx, 9), dtype=cp.float64)

    # Compute term order 1: (1/es^2) * (ex * ux + ey * uy)
    Termorder1 = (1.0 / es**2) * (cp.einsum('q,ij->ijq', ex, u[0]) + cp.einsum('q,ij->ijq', ey, u[1]))

    # Compute squared velocity components
    ux2 = u[0]**2
    uy2 = u[1]**2
    u2 = ux2 + uy2

    # Compute term order 2
    euxy = cp.einsum('q,ij->ijq', ex * ey, u[0] * u[1])
    euxx = cp.einsum('q,ij->ijq', ex**2, u[0]**2)
    euyy = cp.einsum('q,ij->ijq', ey**2, u[1]**2)
    eu2 = 2 * euxy + euxx + euyy
    Termorder2 = (0.5 / es**4) * eu2 - (0.5 / es**2) * u2[:, :, cp.newaxis]

    # Compute feq (equilibrium distribution function)
    feq = cp.einsum('q,ij->ijq', w, rho) * (Termorder2 + Termorder1) + cp.einsum('q,ij->ijq', w, (rho - 1))

    # Apply condition: feq = 0 where nodetype > 0 (solid nodes)
    feq[nodetype > 0] = 0
  

    return feq


# ==========================
# Test Function
# ==========================
def test_compute_edf():
    """Test function for compute_edf."""
    
    # Define test grid size
    ny, nx = 10, 10  # Small grid for testing
    
    # Initialize test arrays
    rho = cp.ones((ny, nx), dtype=cp.float64)  # Uniform density field
    u = cp.zeros((2, ny, nx), dtype=cp.float64)  # Zero velocity field (fluid at rest)
    nodetype = cp.zeros((ny, nx), dtype=cp.float64)  # All nodes are fluid

    # Call the function
    feq = compute_edf(rho, u, nodetype)

    # **Assertions to Validate Results**
    
    # ✅ Check output shape
    assert feq.shape == (ny, nx, 9), f"Expected shape ({ny}, {nx}, 9), but got {feq.shape}"

    # ✅ Check if feq is CuPy array
    assert isinstance(feq, cp.ndarray), "feq is not a CuPy array"

    # ✅ Check that feq is non-negative
    assert cp.all(feq >= 0), "feq contains negative values"

    # ✅ Check sum of feq (should match rho when velocity is zero)
    feq_sum = cp.sum(feq, axis=2)
    assert cp.allclose(feq_sum , rho), "Sum of feq does not match rho"

    # ✅ Check that solid nodes have feq = 0
    nodetype[3, 3] = 1  # Make one solid node
    feq_new = compute_edf(rho, u, nodetype)
    assert cp.all(feq_new[3, 3, :] == 0), "Solid node does not have feq = 0"

    print("✅ All tests passed successfully!")


# ==========================
# Run Test
# ==========================
if __name__ == "__main__":
    test_compute_edf()
