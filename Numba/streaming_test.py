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
                    f[i, j, q] = f_old[per_i, per_j, q + 4]
                else:
                    f[i, j, q] = f_old[per_i, per_j, q - 4]

# Define test grid size
ny, nx = 10, 10

# Define lattice directions
ex = np.array([0, 1, 0, 1, 1, -1, 0, -1, -1])
ey = np.array([0, 0, 1, 1, -1, 0, -1, -1, 1])

# Define nodetype grid (0 for fluid, 1 for solid)
nodetype = np.zeros((ny, nx))

# Define a solid obstacle in the middle
nodetype[4:6, 4:6] = 1

# Initialize distribution functions
f = np.random.rand(ny, nx, 9).astype(np.float64)
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

# Verify streaming (Check if a value moves to its correct neighbor)
streaming_correct = True
for i in range(ny):
    for j in range(nx):
        if nodetype[i, j] == 0:  # Check only fluid nodes
            for q in range(1, 9):
                per_i = (i + ey[q]) % ny
                per_j = (j - ex[q]) % nx
                if nodetype[per_i, per_j] == 0:
                    if not np.isclose(f_result[i, j, q], f_old[per_i, per_j, q]):
                        streaming_correct = False

# Verify bounce-back (Check if solid nodes reflect distributions)
bounceback_correct = True
for i in range(ny):
    for j in range(nx):
        if nodetype[i, j] == 0:  # Check fluid nodes near solids
            for q in range(1, 9):
                per_i = (i + ey[q]) % ny
                per_j = (j - ex[q]) % nx
                if nodetype[per_i, per_j] == 1:  # If neighbor is solid
                    if q < 5:
                        expected = f_old[per_i, per_j, q + 4]
                    else:
                        expected = f_old[per_i, per_j, q - 4]
                    if not np.isclose(f_result[i, j, q], expected):
                        bounceback_correct = False

# Verify periodicity (Check if edges are wrapping correctly)
periodicity_correct = True
for j in range(nx):
    if not np.isclose(f_result[0, j, :], f_result[ny-1, j, :]).all():
        periodicity_correct = False
for i in range(ny):
    if not np.isclose(f_result[i, 0, :], f_result[i, nx-1, :]).all():
        periodicity_correct = False

# Display results
test_results = ([
    ["Streaming Test", streaming_correct],
    ["Bounce-Back Test", bounceback_correct],
    ["Periodicity Test", periodicity_correct]
])

print(test_results)
