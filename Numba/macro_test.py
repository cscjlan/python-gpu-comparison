import numpy as np
from numba import cuda

# Define dtype
dtype = np.float64

# Define a small test case for easier debugging
nx, ny = 4, 4  # Small grid for testing
f_host = np.random.rand(ny, nx, 9).astype(dtype)  # Random values for f
nodetype_host = np.zeros((ny, nx), dtype=dtype)  # All nodes are fluid

# Allocate memory for outputs
rho_host = np.zeros((ny, nx), dtype=dtype)
u_host = np.zeros((2, ny, nx), dtype=dtype)

# Copy data to GPU
f_d = cuda.to_device(f_host)
nodetype_d = cuda.to_device(nodetype_host)
rho_d = cuda.to_device(rho_host)
u_d = cuda.to_device(u_host)


@cuda.jit
def compute_macro_vars_gpu(f, nodetype, rho, u):
    i, j = cuda.grid(2)
    ny, nx = nodetype.shape
    if i < ny and j < nx and nodetype[i, j] <= 0:
        rho[i, j] = f[i, j, 0] + f[i, j, 1] + f[i, j, 2] + f[i, j, 3] + f[i, j, 4] + f[i, j, 5] + f[i, j, 6] + f[i, j, 7] + f[i, j, 8] 
        fdotex = f[i, j, 1] + f[i, j, 3] + f[i, j, 4] - f[i, j, 5] - f[i, j, 7] - f[i, j, 8]
        fdotey = f[i, j, 2] + f[i, j, 3] - f[i, j, 4] - f[i, j, 6] - f[i, j, 7] + f[i, j, 8]
        u[0, i, j] = fdotex / rho[i, j]
        u[1, i, j] = fdotey / rho[i, j]
    elif i < ny and j < nx:
        rho[i, j] = 0
        u[0, i, j] = 0
        u[1, i, j] = 0


# Define CUDA grid and block sizes
threads_per_block = (8, 8)
blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Run the kernel
compute_macro_vars_gpu[blocks_per_grid, threads_per_block](f_d, nodetype_d, rho_d, u_d)

# Copy results back to host
rho_result = rho_d.copy_to_host()
u_result = u_d.copy_to_host()

# Compute expected results on CPU
rho_expected = np.sum(f_host, axis=2)
fdotex_expected = f_host[:, :, 1] + f_host[:, :, 3] + f_host[:, :, 4] - f_host[:, :, 5] - f_host[:, :, 7] - f_host[:, :, 8]
fdotey_expected = f_host[:, :, 2] + f_host[:, :, 3] - f_host[:, :, 4] - f_host[:, :, 6] - f_host[:, :, 7] + f_host[:, :, 8]

# Avoid division by zero
u_expected = np.zeros_like(u_host)
nonzero_mask = rho_expected > 0
u_expected[0, nonzero_mask] = fdotex_expected[nonzero_mask] / rho_expected[nonzero_mask]
u_expected[1, nonzero_mask] = fdotey_expected[nonzero_mask] / rho_expected[nonzero_mask]

# Compare results
assert np.allclose(rho_result, rho_expected), "rho does not match expected values!"
assert np.allclose(u_result, u_expected), "u does not match expected values!"

print("✅ macro var function passed.")

def test_compute_macro_vars():
    nx, ny = 4, 4  # Small grid for testing

    # Test Case 1: All fluid nodes
    f_host = np.random.rand(ny, nx, 9).astype(dtype)  # Random values for f
    nodetype_host = np.zeros((ny, nx), dtype=dtype)  # All nodes fluid

    run_test(f_host, nodetype_host, "All Fluid Nodes")

    # Test Case 2: Some solid nodes
    nodetype_host[1, 1] = 1  # Mark a solid node
    nodetype_host[2, 2] = 1  # Another solid node
    run_test(f_host, nodetype_host, "Some Solid Nodes")

    # Test Case 3: Zero density (all `f = 0`)
    #f_host.fill(0)
    #run_test(f_host, nodetype_host, "Zero Density Case")

    # Test Case 4: Uniform `f`
    f_host.fill(1)
    run_test(f_host, nodetype_host, "Uniform f Distribution")

    # Test Case 5: Edge case with large values
    f_host = np.full((ny, nx, 9), 1e6, dtype=dtype)
    run_test(f_host, nodetype_host, "Large Values in f")

    print("✅ All test cases passed!")

def run_test(f_host, nodetype_host, test_name):
    nx, ny = f_host.shape[:2]

    # Allocate memory for outputs
    rho_host = np.zeros((ny, nx), dtype=dtype)
    u_host = np.zeros((2, ny, nx), dtype=dtype)

    # Copy data to GPU
    f_d = cuda.to_device(f_host)
    nodetype_d = cuda.to_device(nodetype_host)
    rho_d = cuda.to_device(rho_host)
    u_d = cuda.to_device(u_host)

    # Define CUDA grid and block sizes
    threads_per_block = (8, 8)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Run the kernel
    compute_macro_vars_gpu[blocks_per_grid, threads_per_block](f_d, nodetype_d, rho_d, u_d)

    # Copy results back to host
    rho_result = rho_d.copy_to_host()
    u_result = u_d.copy_to_host()

    # Compute expected results on CPU
    rho_expected = np.sum(f_host, axis=2)
    fdotex_expected = f_host[:, :, 1] + f_host[:, :, 3] + f_host[:, :, 4] - f_host[:, :, 5] - f_host[:, :, 7] - f_host[:, :, 8]
    fdotey_expected = f_host[:, :, 2] + f_host[:, :, 3] - f_host[:, :, 4] - f_host[:, :, 6] - f_host[:, :, 7] + f_host[:, :, 8]

    # Avoid division by zero
    u_expected = np.zeros_like(u_host)
    nonzero_mask = rho_expected > 0
    u_expected[0, nonzero_mask] = fdotex_expected[nonzero_mask] / rho_expected[nonzero_mask]
    u_expected[1, nonzero_mask] = fdotey_expected[nonzero_mask] / rho_expected[nonzero_mask]

    # Apply solid node condition (rho = 0, u = 0 where nodetype > 0)
    rho_expected[nodetype_host > 0] = 0
    u_expected[:, nodetype_host > 0] = 0

    # Compare results
    assert np.allclose(rho_result, rho_expected), f"❌ Test '{test_name}' failed: rho mismatch"
    assert np.allclose(u_result, u_expected), f"❌ Test '{test_name}' failed: u mismatch"

    print(f"✅ Test '{test_name}' passed!")

# Run all tests
test_compute_macro_vars()


