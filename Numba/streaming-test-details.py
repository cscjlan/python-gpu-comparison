import numpy as np
from numba import cuda
#import pandas as pd


@cuda.jit
def stream_and_bounce_gpu(f, f_old, nodetype, ex, ey):
    i, j = cuda.grid(2)
    ny, nx = nodetype.shape

    if i < ny and j < nx and nodetype[i, j] <= 0:
        for q in range(1, 9):  # Direction 0 does not need streaming
            per_i = (i + int(ey[q])) % ny  # Periodic boundary
            per_j = (j - int(ex[q])) % nx  # Periodic boundary
            
            if nodetype[per_i, per_j] <= 0:
                f[i, j, q] = f_old[per_i, per_j, q]  # Normal streaming
            else:  # Bounce-back condition
                if q < 5:
                    f[i, j, q] = f_old[i, j, q + 4]
                else:
                    f[i, j, q] = f_old[i, j, q - 4]


# Define test grid size
# Define grid size
ny, nx = 4, 4  # 4x4 grid, total 16 grids

# Initialize f with structured values
f = np.zeros((ny, nx, 9), dtype=np.float64)

# Assign values based on grid index
for i in range(ny):
    for j in range(nx):
        grid_index = i * nx + j  # Unique grid index from 0 to 15
        f[i, j, :] = np.linspace(grid_index, grid_index + 0.8, 9)

# Display the generated f array
print("Generated f array:\n", f)

# Define lattice directions
ex = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1])
ey = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1])

# Define nodetype grid (0 for fluid, 1 for solid)
nodetype = np.zeros((ny, nx))

# Define a solid obstacle in the middle
nodetype[2, 2] = 1

# Initialize distribution functions
#f = np.random.rand(ny, nx, 9).astype(np.float64)
f_old = np.copy(f)

# Allocate GPU memory
d_f = cuda.to_device(f)
d_f_old = cuda.to_device(f_old)
d_nodetype = cuda.to_device(nodetype)
d_ex = cuda.to_device(ex)
d_ey = cuda.to_device(ey)

# Define kernel launch configuration
threads_per_block = (16, 16)
blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch the kernel
stream_and_bounce_gpu[blocks_per_grid, threads_per_block](d_f, d_f_old, d_nodetype, d_ex, d_ey)

# Copy result back to host
f_result = d_f.copy_to_host()


print("Result f array:\n", f_result)