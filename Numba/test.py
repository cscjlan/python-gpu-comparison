import numpy as np
from numba import cuda

@cuda.jit
def add_kernel(d_arr, val):
    idx = cuda.grid(1)  # Get the absolute thread index
    if idx < d_arr.size:  # Ensure within bounds
        d_arr[idx] += val

# Initialize data
arr = np.arange(100, dtype=np.float32)
d_arr = cuda.to_device(arr)  # Copy data to GPU

# Define grid and block size
threads_per_block = 128
blocks_per_grid = (arr.size + threads_per_block - 1) // threads_per_block

# Run the kernel
add_kernel[blocks_per_grid, threads_per_block](d_arr, 5.0)

# Copy result back to CPU
result = d_arr.copy_to_host()
print(result)



