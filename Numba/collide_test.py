import numpy as np
from numba import cuda

# Define dtype
dtype = np.float64


@cuda.jit
def collide_gpu(f, rho, u, nodetype, tau, Fg):
    i, j = cuda.grid(2)
    ny, nx = nodetype.shape
    if i < ny and j < nx and nodetype[i, j] <= 0:
        # Apply forcing
        u[0, i, j] += Fg[0, i, j] * tau[i, j] / rho[i, j]
        u[1, i, j] += Fg[1, i, j] * tau[i, j] / rho[i, j]

        # Compute equilibrium distribution function explicitly
        feq0 = rho[i, j] * (-2.0/3.0 * u[0, i, j]**2 - 2.0/3.0 * u[1, i, j]**2 + 4.0/9.0)
        feq1 = rho[i, j] * ((1.0/3.0) * u[0, i, j]**2 + (1.0/3.0) * u[0, i, j] - 1.0/6.0 * u[1, i, j]**2 + 1.0/9.0)
        feq2 = rho[i, j] * (-1.0/6.0 * u[0, i, j]**2 + (1.0/3.0) * u[1, i, j]**2 + (1.0/3.0) * u[1, i, j] + 1.0/9.0)
        feq3 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 + (1.0/12.0) * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 + 
                            (1.0/12.0) * u[1, i, j] + (1.0/8.0) * (u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)
        feq4 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 + (1.0/12.0) * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 - 
                            1.0/12.0 * u[1, i, j] + (1.0/8.0) * (u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
        feq5 = rho[i, j] * ((1.0/3.0) * u[0, i, j]**2 - 1.0/3.0 * u[0, i, j] - 1.0/6.0 * u[1, i, j]**2 + 1.0/9.0)
        feq6 = rho[i, j] * (-1.0/6.0 * u[0, i, j]**2 + (1.0/3.0) * u[1, i, j]**2 - 1.0/3.0 * u[1, i, j] + 1.0/9.0)
        feq7 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/12.0 * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 - 
                            1.0/12.0 * u[1, i, j] + (1.0/8.0) * (-u[0, i, j] - u[1, i, j])**2 + 1.0/36.0)
        feq8 = rho[i, j] * (-1.0/24.0 * u[0, i, j]**2 - 1.0/12.0 * u[0, i, j] - 1.0/24.0 * u[1, i, j]**2 + 
                            (1.0/12.0) * u[1, i, j] + (1.0/8.0) * (-u[0, i, j] + u[1, i, j])**2 + 1.0/36.0)

        # Collision step
        f[i, j, 0] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 0] + (1.0 / tau[i, j]) * feq0
        f[i, j, 1] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 1] + (1.0 / tau[i, j]) * feq1
        f[i, j, 2] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 2] + (1.0 / tau[i, j]) * feq2
        f[i, j, 3] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 3] + (1.0 / tau[i, j]) * feq3
        f[i, j, 4] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 4] + (1.0 / tau[i, j]) * feq4
        f[i, j, 5] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 5] + (1.0 / tau[i, j]) * feq5
        f[i, j, 6] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 6] + (1.0 / tau[i, j]) * feq6
        f[i, j, 7] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 7] + (1.0 / tau[i, j]) * feq7
        f[i, j, 8] = (1.0 - (1.0 / tau[i, j])) * f[i, j, 8] + (1.0 / tau[i, j]) * feq8


def test_collide_gpu_extended():
    nx, ny = 4, 4  # Small test grid

    # Define parameters
    rho_host = np.random.rand(ny, nx).astype(dtype)
    u_host = np.random.rand(2, ny, nx).astype(dtype)
    tau_host = np.ones((ny, nx), dtype=dtype)  # Uniform relaxation time
    Fg_host = np.zeros((2, ny, nx), dtype=dtype)  # No external force
    nodetype_host = np.zeros((ny, nx), dtype=dtype)  # All fluid
    f_host = np.random.rand(ny, nx, 9).astype(dtype)  # Random initial f

    # Test 1: All Fluid Nodes (Already Passed ✅)
    run_test_collide(f_host, rho_host, u_host, nodetype_host, tau_host, Fg_host, "All Fluid Nodes")

    # Test 2: Some Solid Nodes
    nodetype_host[1, 1] = 1  # Mark a solid node
    nodetype_host[2, 2] = 1
    run_test_collide(f_host, rho_host, u_host, nodetype_host, tau_host, Fg_host, "Some Solid Nodes")

    # Test 3: Zero Density (`rho = 0`)
    #rho_host.fill(0)
    #run_test_collide(f_host, rho_host, u_host, nodetype_host, tau_host, Fg_host, "Zero Density Case")

    # Test 4: Zero Velocity (`u = 0`)
    rho_host = np.random.rand(ny, nx).astype(dtype)
    u_host.fill(0)
    run_test_collide(f_host, rho_host, u_host, nodetype_host, tau_host, Fg_host, "Zero Velocity Case")

    # Test 5: Uniform `rho` and `u`
    rho_host.fill(1)
    u_host.fill(1)
    run_test_collide(f_host, rho_host, u_host, nodetype_host, tau_host, Fg_host, "Uniform rho and u")

    # Test 6: Large Values in `rho` and `u`
    rho_host.fill(1e6)
    u_host.fill(1e6)
    run_test_collide(f_host, rho_host, u_host, nodetype_host, tau_host, Fg_host, "Large Values in rho and u")

    print("✅ All extended test cases passed!")

def run_test_collide(f_host, rho_host, u_host, nodetype_host, tau_host, Fg_host, test_name):
    nx, ny = rho_host.shape

    # Copy data to GPU
    f_d = cuda.to_device(f_host)
    rho_d = cuda.to_device(rho_host)
    u_d = cuda.to_device(u_host)
    nodetype_d = cuda.to_device(nodetype_host)
    tau_d = cuda.to_device(tau_host)
    Fg_d = cuda.to_device(Fg_host)

    # Define CUDA grid and block sizes
    threads_per_block = (8, 8)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Run the kernel
    collide_gpu[blocks_per_grid, threads_per_block](f_d, rho_d, u_d, nodetype_d, tau_d, Fg_d)

    # Copy results back to host
    f_result = f_d.copy_to_host()

    # Compute expected results on CPU
    f_expected = np.copy(f_host)

    for i in range(ny):
        for j in range(nx):
            if nodetype_host[i, j] <= 0:  # Fluid node condition
                # Apply forcing
                u_host[0, i, j] += Fg_host[0, i, j] * tau_host[i, j] / rho_host[i, j]
                u_host[1, i, j] += Fg_host[1, i, j] * tau_host[i, j] / rho_host[i, j]

                # Compute equilibrium distribution function explicitly
                feq = np.zeros(9, dtype=dtype)
                feq[0] = rho_host[i, j] * (-2.0/3.0 * u_host[0, i, j]**2 - 2.0/3.0 * u_host[1, i, j]**2 + 4.0/9.0)
                feq[1] = rho_host[i, j] * ((1.0/3.0) * u_host[0, i, j]**2 + (1.0/3.0) * u_host[0, i, j] - 1.0/6.0 * u_host[1, i, j]**2 + 1.0/9.0)
                feq[2] = rho_host[i, j] * (-1.0/6.0 * u_host[0, i, j]**2 + (1.0/3.0) * u_host[1, i, j]**2 + (1.0/3.0) * u_host[1, i, j] + 1.0/9.0)
                feq[3] = rho_host[i, j] * (-1.0/24.0 * u_host[0, i, j]**2 + (1.0/12.0) * u_host[0, i, j] - 1.0/24.0 * u_host[1, i, j]**2 + 
                                          (1.0/12.0) * u_host[1, i, j] + (1.0/8.0) * (u_host[0, i, j] + u_host[1, i, j])**2 + 1.0/36.0)
                feq[4]= rho_host[i, j]*(-1.0/24.0*u_host[0,i, j]**2 + (1.0/12.0)*u_host[0,i, j] - 1.0/24.0*u_host[1,i,j]**2 - 
                1.0/12.0*u_host[1,i,j] + (1.0/8.0)*(u_host[0,i,j] - u_host[1,i,j])**2 + 1.0/36.0)
                feq[5]= rho_host[i,j]*((1.0/3.0)*u_host[0,i,j]**2 - 1.0/3.0*u_host[0,i,j] - 1.0/6.0*u_host[1,i,j]**2 + 1.0/9.0)
                feq[6]= rho_host[i,j]*(-1.0/6.0*u_host[0,i,j]**2 + (1.0/3.0)*u_host[1,i,j]**2 - 1.0/3.0*u_host[1,i,j] + 1.0/9.0)
                feq[7]= rho_host[i,j]*(-1.0/24.0*u_host[0,i,j]**2 - 1.0/12.0*u_host[0,i,j] - 1.0/24.0*u_host[1,i,j]**2 - 
                1.0/12.0*u_host[1,i,j] + (1.0/8.0)*(-u_host[0,i,j] - u_host[1,i,j])**2 + 1.0/36.0)
                feq[8]= rho_host[i,j]*(-1.0/24.0*u_host[0,i,j]**2 - 1.0/12.0*u_host[0,i,j] - 1.0/24.0*u_host[1,i,j]**2 + ( 
                1.0/12.0)*u_host[1,i,j] + (1.0/8.0)*(-u_host[0,i,j] + u_host[1,i,j])**2 + 1.0/36.0)
                
                # Collision step
                for q in range(9):
                    f_expected[i, j, q] = (1.0 - (1.0 / tau_host[i, j])) * f_host[i, j, q] + (1.0 / tau_host[i, j]) * feq[q]

            else:
                f_expected[i, j, :] = f_host[i, j, :]  # No change for solid nodes

    # Compare results
    assert np.allclose(f_result, f_expected, atol=1e-6), f"❌ Test '{test_name}' failed: f mismatch!"

    print(f"✅ Test '{test_name}' passed!")

# Run the extended tests
test_collide_gpu_extended()
