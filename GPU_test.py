from numba import cuda

def main():
    print("Checking CUDA availability...")
    if not cuda.is_available():
        print("CUDA is not available. Please check your setup.")
        return

    print("CUDA is available!")
    device = cuda.get_current_device()
    print(f"Device Name: {device.name}")

    # Get memory information using the current CUDA context
    mem_info = cuda.current_context().get_memory_info()
    print(f"Total Memory: {mem_info[1] / 1e9:.2f} GB")  # Convert bytes to GB
    print(f"Free Memory: {mem_info[0] / 1e9:.2f} GB")   # Convert bytes to GB

    print(f"Compute Capability: {device.compute_capability}")

if __name__ == "__main__":
    main()
