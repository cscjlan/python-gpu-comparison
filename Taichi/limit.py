import taichi as ti

ti.init(arch=ti.gpu, device_memory_fraction=0.3)  # Use 90% of VRAM

# Define the sparse field
f = ti.field(dtype=ti.f32)

# Use a hierarchical structure to break down the domain into smaller parts
block_size = 16  # Choose a reasonable block size (16x16x16)
root = ti.root.pointer(ti.ijk, (800 // block_size, 800 // block_size, 800 // block_size))  # High-level division
root.dense(ti.ijk, (block_size, block_size, block_size)).dense(ti.l, 19).place(f)  # Each block contains full 19 velocity values

# Function to initialize data
@ti.kernel
def initialize():
    for i, j, k in ti.ndrange(800, 800, 800):
        if (i + j + k) % 20 == 0:  # Reduce active blocks to avoid OOM
            for q in range(19):  
                f[i, j, k, q] = i * j * k * 0.00001  # Assign value

# Run initialization
initialize()
print(f.shape)
print("Sparse field initialized successfully!")
