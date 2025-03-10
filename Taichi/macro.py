import numpy as np
import numba as nb
from numba import cuda
import taichi as ti
import time
import matplotlib.pyplot as plt

dtype = np.float64

@cuda.jit
def compute_macro_vars_gpu(f, nodetype, rho, u):
    j, i = cuda.grid(2)
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


# Taichi GPU Setup
ti.init(arch=ti.gpu)

@ti.kernel
def compute_macro_vars_taichi(f: ti.types.ndarray(), nodetype: ti.types.ndarray(), rho: ti.types.ndarray(), u: ti.types.ndarray()):
    for i, j in ti.ndrange(f.shape[0], f.shape[1]):
        if nodetype[i, j] <= 0:
            rho_ij = f[i, j, 0] + f[i, j, 1] + f[i, j, 2] + f[i, j, 3] + f[i, j, 4] + f[i, j, 5] + f[i, j, 6] + f[i, j, 7] + f[i, j, 8]
            fdotex = f[i, j, 1] + f[i, j, 3] + f[i, j, 4] - f[i, j, 5] - f[i, j, 7] - f[i, j, 8]
            fdotey = f[i, j, 2] + f[i, j, 3] - f[i, j, 4] - f[i, j, 6] - f[i, j, 7] + f[i, j, 8]
            rho[i, j] = rho_ij
            u[0, i, j] = fdotex / rho_ij
            u[1, i, j] = fdotey / rho_ij
        else:
            rho[i, j] = 0
            u[0, i, j] = 0
            u[1, i, j] = 0


def test_macro():
    nx = 1000  # Adjusted for reasonable execution time
    ny = 5000  
    rho = np.ones((ny, nx), dtype=dtype)
    u = np.ones((2, ny, nx), dtype=dtype)*1e-4
    nodetype = np.zeros((ny, nx), dtype=dtype)
    nodetype[0, :] = 1
    nodetype[-1, :] = 1
    f = np.random.rand(ny, nx, 9).astype(dtype)

    # ======= Numba GPU Execution =======
    f_d = cuda.to_device(f)
    rho_d = cuda.to_device(rho)
    u_d = cuda.to_device(u)
    nodetype_d = cuda.to_device(nodetype)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start = time.time()
    compute_macro_vars_gpu[blocks_per_grid, threads_per_block](f_d, nodetype_d, rho_d, u_d)
    cuda.synchronize()
    end = time.time()
    
    rho_gpu = rho_d.copy_to_host()
    u_gpu = u_d.copy_to_host()
    print(f"Numba GPU Execution Time: {end - start:.6f} sec")

    # ======= Taichi GPU Execution =======
    f_taichi = ti.ndarray(dtype=ti.f64, shape=(ny, nx, 9))
    nodetype_taichi = ti.ndarray(dtype=ti.f64, shape=(ny, nx))
    rho_taichi = ti.ndarray(dtype=ti.f64, shape=(ny, nx))
    u_taichi = ti.ndarray(dtype=ti.f64, shape=(2, ny, nx))

    f_taichi.from_numpy(f)
    nodetype_taichi.from_numpy(nodetype)

    start = time.time()
    compute_macro_vars_taichi(f_taichi, nodetype_taichi, rho_taichi, u_taichi)
    ti.sync()
    end = time.time()

    rho_taichi_np = rho_taichi.to_numpy()
    u_taichi_np = u_taichi.to_numpy()
    print(f"Taichi GPU Execution Time: {end - start:.6f} sec")

    # ======= Accuracy Check =======
    rho_match = np.allclose(rho_gpu, rho_taichi_np, atol=1e-8)
    u_match = np.allclose(u_gpu, u_taichi_np, atol=1e-8)

    print(f"Density Match: {rho_match}")
    print(f"Velocity Match: {u_match}")


test_macro()










# import numpy as np
# import numba as nb
# from numba import cuda
# import matplotlib.pyplot as plt
# import time
# dtype = np.float64



# dtype = np.float64



# @cuda.jit
# def compute_macro_vars_gpu(f, nodetype, rho, u):
#     j, i = cuda.grid(2)
#     ny, nx = nodetype.shape
#     if i < ny and j < nx and nodetype[i, j] <= 0:
#         rho[i, j] = f[i, j, 0] + f[i, j, 1] + f[i, j, 2] + f[i, j, 3] + f[i, j, 4] + f[i, j, 5] + f[i, j, 6] + f[i, j, 7] + f[i, j, 8] 
#         fdotex = f[i, j, 1] + f[i, j, 3] + f[i, j, 4] - f[i, j, 5] - f[i, j, 7] - f[i, j, 8]
#         fdotey = f[i, j, 2] + f[i, j, 3] - f[i, j, 4] - f[i, j, 6] - f[i, j, 7] + f[i, j, 8]
#         u[0, i, j] = fdotex / rho[i, j]
#         u[1, i, j] = fdotey / rho[i, j]
#     elif i < ny and j < nx:
#         rho[i, j] = 0
#         u[0, i, j] = 0
#         u[1, i, j] = 0


# def test_macro():
#     nx = 10000
#     ny = 10000
#     rho = np.ones((ny, nx), dtype=dtype)
#     u = np.ones((2, ny, nx), dtype=dtype)*1e-4
#     nodetype = np.zeros((ny, nx), dtype=dtype)
#     nodetype[0, :] = 1
#     nodetype[-1, :] = 1
#     f = np.random.rand(ny, nx, 9).astype(dtype)

#     # GPU Memory Allocation
#     f_d = cuda.to_device(f)
#     rho_d = cuda.to_device(rho)
#     u_d = cuda.to_device(u)
#     nodetype_d = cuda.to_device(nodetype)

#     #f[10,10,3] +=1e-14

#     threads_per_block = (16, 16)
#     blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
#     blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
#     blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

#     compute_macro_vars_gpu[blocks_per_grid, threads_per_block](f_d, nodetype_d, rho_d, u_d)


#     rho_gpu = rho_d.copy_to_host()
#     u_gpu = u_d.copy_to_host()


#     #f_cpu[25] = 0

#     # equal = np.allclose(rho_gpu, rho, atol=1e-17, rtol=0)
#     # equal2 = np.allclose(u_gpu, u, atol=1e-17, rtol=0)
#     # print(equal,equal2)

# test_macro()
