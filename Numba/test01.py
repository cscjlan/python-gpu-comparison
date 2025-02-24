import numpy as np
from numba import cuda

@cuda.jit
def add_kernel_2d(d_arr, val):
    row, col = cuda.grid(2)  # Get the thread indices for 2D
    if row < d_arr.shape[0] and col < d_arr.shape[1]:  # Ensure within bounds
        d_arr[row, col] += val

# Initialize a 2D array
arr = np.arange(12, dtype=np.float32).reshape(3, 4)  # Shape (3, 4)
d_arr = cuda.to_device(arr)  # Copy data to GPU

# Define grid and block size for 2D
threads_per_block = (16, 16)  # 16x16 block
blocks_per_grid_x = (arr.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (arr.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Run the kernel
add_kernel_2d[blocks_per_grid, threads_per_block](d_arr, 5.0)

# Copy result back to CPU
result = d_arr.copy_to_host()
print(result)
