import numpy as np
from numba import cuda

# Define dtype
dtype = np.float64


@cuda.jit
def compute_edf_gpu(rho, u, nodetype, feq,ex,ey,w):
    i, j = cuda.grid(2)
    ny, nx = nodetype.shape
    if i < ny and j < nx and nodetype[i, j] <= 0:
        for q in range(9):
            es= (1/3)**0.5
            Termorder1 = (1. / es**2) * (ex[q] * u[0, i, j] + ey[q] * u[1, i, j])
            euxy = ex[q] * ey[q] * u[0, i, j] * u[1, i, j]
            euxx = ex[q] * ex[q] * u[0, i, j] * u[0, i, j]
            euyy = ey[q] * ey[q] * u[1, i, j] * u[1, i, j]
            eu2 = 2 * euxy + euxx + euyy
            ux2 = u[0, i, j] * u[0, i, j]
            uy2 = u[1, i, j] * u[1, i, j]
            u2 = ux2 + uy2
            Termorder2 = (0.5 / es**4) * eu2 - (0.5 / es**2) * u2
            feq[i, j, q] = w[q] * rho[i, j] * (1 + Termorder1 + Termorder2)

def test_compute_edf():
    nx, ny = 4, 4  # Small test grid

    # Define parameters
    ex_host = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
    ey_host = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
    w_host = np.array([4/9, 1/9, 1/9, 1/36, 1/36, 1/9, 1/9, 1/36, 1/36], dtype=dtype)

    rho_host = np.random.rand(ny, nx).astype(dtype)  # Random density values
    u_host = np.random.rand(2, ny, nx).astype(dtype)  # Random velocity components
    nodetype_host = np.zeros((ny, nx), dtype=dtype)  # All fluid nodes
    feq_host = np.zeros((ny, nx, 9), dtype=dtype)  # Empty equilibrium function

    # Copy data to GPU
    ex_d = cuda.to_device(ex_host)
    ey_d = cuda.to_device(ey_host)
    w_d = cuda.to_device(w_host)
    rho_d = cuda.to_device(rho_host)
    u_d = cuda.to_device(u_host)
    nodetype_d = cuda.to_device(nodetype_host)
    feq_d = cuda.to_device(feq_host)

    # Define CUDA grid and block sizes
    threads_per_block = (8, 8)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Run the kernel
    compute_edf_gpu[blocks_per_grid, threads_per_block](rho_d, u_d, nodetype_d, feq_d, ex_d, ey_d, w_d)

    # Copy results back to host
    feq_result = feq_d.copy_to_host()

    # Compute expected results on CPU
    feq_expected = np.zeros((ny, nx, 9), dtype=dtype)
    es = (1/3)**0.5  # Define es

    for i in range(ny):
        for j in range(nx):
            if nodetype_host[i, j] <= 0:  # Fluid node condition
                for q in range(9):
                    Termorder1 = (1. / es**2) * (ex_host[q] * u_host[0, i, j] + ey_host[q] * u_host[1, i, j])
                    euxy = ex_host[q] * ey_host[q] * u_host[0, i, j] * u_host[1, i, j]
                    euxx = ex_host[q] * ex_host[q] * u_host[0, i, j] * u_host[0, i, j]
                    euyy = ey_host[q] * ey_host[q] * u_host[1, i, j] * u_host[1, i, j]
                    eu2 = 2 * euxy + euxx + euyy
                    ux2 = u_host[0, i, j] * u_host[0, i, j]
                    uy2 = u_host[1, i, j] * u_host[1, i, j]
                    u2 = ux2 + uy2
                    Termorder2 = (0.5 / es**4) * eu2 - (0.5 / es**2) * u2
                    feq_expected[i, j, q] = w_host[q] * rho_host[i, j] * (1 + Termorder1 + Termorder2)

    # Compare results
    assert np.allclose(feq_result, feq_expected, atol=1e-6), "❌ Test failed: feq mismatch!"

    print("✅ edf function passed.")

# Run the test
test_compute_edf()


def test_compute_edf_extended():
    nx, ny = 4, 4  # Small test grid

    # Define parameters
    ex_host = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
    ey_host = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
    w_host = np.array([4/9, 1/9, 1/9, 1/36, 1/36, 1/9, 1/9, 1/36, 1/36], dtype=dtype)

    # Test 1: All Fluid Nodes (Already Passed ✅)
    rho_host = np.random.rand(ny, nx).astype(dtype)
    u_host = np.random.rand(2, ny, nx).astype(dtype)
    nodetype_host = np.zeros((ny, nx), dtype=dtype)  # All fluid
    run_test_compute_edf(rho_host, u_host, nodetype_host, ex_host, ey_host, w_host, "All Fluid Nodes")

    # Test 2: Some Solid Nodes
    nodetype_host[1, 1] = 1  # Mark a solid node
    nodetype_host[2, 2] = 1
    run_test_compute_edf(rho_host, u_host, nodetype_host, ex_host, ey_host, w_host, "Some Solid Nodes")

    # Test 3: Zero Density (`rho = 0`)
    #rho_host.fill(0)
    #run_test_compute_edf(rho_host, u_host, nodetype_host, ex_host, ey_host, w_host, "Zero Density Case")

    # Test 4: Zero Velocity (`u = 0`)
    rho_host = np.random.rand(ny, nx).astype(dtype)
    u_host.fill(0)
    run_test_compute_edf(rho_host, u_host, nodetype_host, ex_host, ey_host, w_host, "Zero Velocity Case")

    # Test 5: Uniform `rho` and `u`
    rho_host.fill(1)
    u_host.fill(1)
    run_test_compute_edf(rho_host, u_host, nodetype_host, ex_host, ey_host, w_host, "Uniform rho and u")

    # Test 6: Large Values in `rho` and `u`
    rho_host.fill(1e6)
    u_host.fill(1e6)
    run_test_compute_edf(rho_host, u_host, nodetype_host, ex_host, ey_host, w_host, "Large Values in rho and u")

    print("✅ All extended test cases passed!")

def run_test_compute_edf(rho_host, u_host, nodetype_host, ex_host, ey_host, w_host, test_name):
    nx, ny = rho_host.shape

    # Allocate memory for outputs
    feq_host = np.zeros((ny, nx, 9), dtype=dtype)

    # Copy data to GPU
    ex_d = cuda.to_device(ex_host)
    ey_d = cuda.to_device(ey_host)
    w_d = cuda.to_device(w_host)
    rho_d = cuda.to_device(rho_host)
    u_d = cuda.to_device(u_host)
    nodetype_d = cuda.to_device(nodetype_host)
    feq_d = cuda.to_device(feq_host)

    # Define CUDA grid and block sizes
    threads_per_block = (8, 8)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Run the kernel
    compute_edf_gpu[blocks_per_grid, threads_per_block](rho_d, u_d, nodetype_d, feq_d, ex_d, ey_d, w_d)

    # Copy results back to host
    feq_result = feq_d.copy_to_host()

    # Compute expected results on CPU
    feq_expected = np.zeros((ny, nx, 9), dtype=dtype)
    es = (1/3)**0.5  # Define es

    for i in range(ny):
        for j in range(nx):
            if nodetype_host[i, j] <= 0:  # Fluid node condition
                for q in range(9):
                    Termorder1 = (1. / es**2) * (ex_host[q] * u_host[0, i, j] + ey_host[q] * u_host[1, i, j])
                    euxy = ex_host[q] * ey_host[q] * u_host[0, i, j] * u_host[1, i, j]
                    euxx = ex_host[q] * ex_host[q] * u_host[0, i, j] * u_host[0, i, j]
                    euyy = ey_host[q] * ey_host[q] * u_host[1, i, j] * u_host[1, i, j]
                    eu2 = 2 * euxy + euxx + euyy
                    ux2 = u_host[0, i, j] * u_host[0, i, j]
                    uy2 = u_host[1, i, j] * u_host[1, i, j]
                    u2 = ux2 + uy2
                    Termorder2 = (0.5 / es**4) * eu2 - (0.5 / es**2) * u2
                    feq_expected[i, j, q] = w_host[q] * rho_host[i, j] * (1 + Termorder1 + Termorder2)
            else:
                feq_expected[i, j, :] = 0  # Solid nodes should remain 0

    # Compare results
    assert np.allclose(feq_result, feq_expected, atol=1e-6), f"❌ Test '{test_name}' failed: feq mismatch!"

    print(f"✅ Test '{test_name}' passed!")

# Run the extended tests
test_compute_edf_extended()
