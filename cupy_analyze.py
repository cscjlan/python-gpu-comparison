import cupy as cp
import matplotlib.pyplot as plt
import time

dtype = cp.float32

# LBM constants
ex = cp.array([0, 1, 0, 1, 1, -1, 0, -1, -1], dtype=dtype)
ey = cp.array([0, 0, 1, 1, -1, 0, -1, -1, 1], dtype=dtype)
es = cp.sqrt(1 / 3)
w0, ws, wl = 4 / 9, 1 / 9, 1 / 36
w = cp.array([w0, ws, ws, wl, wl, ws, ws, wl, wl], dtype=dtype)


# ============ PROFILING FUNCTION ============ #
def profile_function(func, *args):
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()  # Start timing
    result = func(*args)  # Call function
    end_event.record()  # End timing
    end_event.synchronize()

    elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)  # Time in ms
    print(f"{func.__name__} execution time: {elapsed_time:.3f} ms")

    return result


# ============ LBM FUNCTIONS ============ #
def compute_macro_vars(f, nodetype, rho, u):
    mask = nodetype <= 0  # Shape: (ny, nx)
    
    rho[mask] = cp.sum(f[mask, :], axis=1)  # Sum across velocity directions

    # Compute velocity components
    fdotex = f[:, :, 1] + f[:, :, 3] + f[:, :, 4] - f[:, :, 5] - f[:, :, 7] - f[:, :, 8]
    fdotey = f[:, :, 2] + f[:, :, 3] - f[:, :, 4] - f[:, :, 6] - f[:, :, 7] + f[:, :, 8]

    # Apply mask correctly by ensuring it has the same shape
    u[0][mask] = fdotex[mask] / rho[mask]
    u[1][mask] = fdotey[mask] / rho[mask]


def compute_edf(rho, u, nodetype):
    feq = cp.zeros_like(rho[..., None], dtype=dtype).repeat(9, axis=-1)
    mask = nodetype <= 0
    for q in range(9):
        Termorder1 = (1. / es ** 2) * (ex[q] * u[0] + ey[q] * u[1])
        euxy = ex[q] * ey[q] * u[0] * u[1]
        euxx = ex[q] ** 2 * u[0] ** 2
        euyy = ey[q] ** 2 * u[1] ** 2
        eu2 = 2 * euxy + euxx + euyy
        ux2 = u[0] ** 2
        uy2 = u[1] ** 2
        u2 = ux2 + uy2
        Termorder2 = (0.5 / es ** 4) * eu2 - (0.5 / es ** 2) * u2
        feq[..., q] = w[q] * rho * (1 + Termorder1 + Termorder2)
    feq[~mask] = 0  # Set to 0 for non-fluid nodes
    return feq


def collide(f, rho, u, nodetype, tau, Fg):
    mask = nodetype <= 0
    u[0][mask] += (Fg[0][mask] * tau[mask]) / rho[mask]
    u[1][mask] += (Fg[1][mask] * tau[mask]) / rho[mask]
    
    feq = compute_edf(rho, u, nodetype)
    
    f[mask] = (1.0 - (1.0 / tau[mask, None])) * f[mask] + (1.0 / tau[mask, None]) * feq[mask]
    
    # Swap streaming directions
    for q in range(1, 5):
        f[..., q], f[..., q + 4] = f[..., q + 4], f[..., q]


def stream_and_bounce(f, nodetype):
    ny, nx = nodetype.shape
    f_next = cp.zeros_like(f)
    mask = nodetype <= 0  # Boolean mask for fluid nodes

    # Apply streaming
    for q in range(1, 5):
        nexti = (cp.arange(ny)[:, None] - ey[q]).clip(0, ny - 1).astype(cp.int32)
        nextj = (cp.arange(nx)[None, :] + ex[q]).clip(0, nx - 1).astype(cp.int32)

        valid_mask = mask[nexti, nextj]  # Ensure indices are integers
        f_next[valid_mask, q] = f[valid_mask, q + 4]
        f_next[valid_mask, q + 4] = f[valid_mask, q]

    f[:] = f_next


# ============ MAIN FUNCTION ============ #
def test_lb():
    nx, ny, niters = 200, 200, 10000
    rho = cp.ones((ny, nx), dtype=dtype)
    tau = cp.ones((ny, nx), dtype=dtype)
    u = cp.zeros((2, ny, nx), dtype=dtype)
    Fg = cp.zeros((2, ny, nx), dtype=dtype)
    Fg[0, :, :] = 1e-4
    nodetype = cp.zeros((ny, nx), dtype=dtype)
    nodetype[0, :] = 1
    nodetype[-1, :] = 1

    # Profile initial EDF computation
    f = profile_function(compute_edf, rho, u, nodetype)

    # Profile full simulation loop
    t0 = time.time()
    for _ in range(niters):
        profile_function(collide, f, rho, u, nodetype, tau, Fg)
        profile_function(stream_and_bounce, f, nodetype)
        profile_function(compute_macro_vars, f, nodetype, rho, u)
    t1 = time.time()

    mlups = (ny * nx * niters * 1e-6) / (t1 - t0)
    print("\n=== PERFORMANCE SUMMARY ===")
    print("MLUPS:", mlups)
    print("Total Time taken:", t1 - t0, "seconds")

    # Transfer results to CPU for plotting
    u_cpu = cp.asnumpy(u)

    plt.figure(1)
    plt.quiver(u_cpu[0], u_cpu[1])
    plt.figure(2)
    plt.plot(u_cpu[0][:, nx // 2])
    plt.figure(3)
    plt.imshow(u_cpu[0])
    plt.show()


# Run the test
test_lb()
