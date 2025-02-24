import cupy as cp
import matplotlib.pyplot as plt

import time  # Import at the top of your script

start_time = time.time()  # Record start time

def compute_macro_quantities(f, obstacles, ex, ey):
    rho = f.sum(0)
    fieix = f * ex[:, None, None]
    ux = (rho / (1e-30 + rho) ** 2) * fieix.sum(0)
    fieiy = f * ey[:, None, None]
    uy = (rho / (1e-30 + rho) ** 2) * fieiy.sum(0)
    
    # Set solid node fluxes and density to zero
    mask = (obstacles <= 0).astype(cp.float32)
    rho *= mask
    ux *= mask
    uy *= mask
    
    return rho, ux, uy


def equillibrium(f, rho, ux, uy, obstacles, ex, ey, es, w):
    Termorder1  = (1. / es**2) * (ex[:, None, None] * ux + ey[:, None, None] * uy)
    euxy        = ex[:, None, None] * ey[:, None, None] * ux * uy
    euxx        = ex[:, None, None] * ex[:, None, None] * ux * ux
    euyy        = ey[:, None, None] * ey[:, None, None] * uy * uy
    eu2         = 2. * euxy + euxx + euyy
    u2          = ux**2 + uy**2
    Termorder2  = (0.5 / es**4) * eu2 - (0.5 / es**2) * u2
    feq         = w[:, None, None] * rho * (1 + Termorder1 + Termorder2)
    
    return feq


def collision(f, rho, ux, uy, obstacles, ex, ey, es, w, omega=cp.float32(1.0)):
    feq = equillibrium(f, rho, ux, uy, obstacles, ex, ey, es, w)
    f = (1. - omega) * f + omega * feq
    f *= (obstacles <= 0).astype(cp.float32)  # Mask solid nodes
    return f


def propogation(f, ex, ey, obstacles):
    ex = ex.astype(cp.int32)
    ey = ey.astype(cp.int32)
    solid_node_in_path = cp.zeros(f.shape, dtype=cp.float32)
    fc_bb = cp.zeros(f.shape, dtype=cp.float32)
    solid_node_exist = (obstacles > 0).astype(cp.float32)
    idx = cp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=cp.int32)        

    # See if solid node exists on propagation path
    for j in range(9):
        solid_node_in_path[idx[j]] = solid_node_exist.copy()
        solid_node_in_path[idx[j]] = cp.roll(solid_node_in_path[idx[j]], -ex[j], axis=1)
        solid_node_in_path[idx[j]] = cp.roll(solid_node_in_path[idx[j]], ey[j], axis=0)    

    f_propogation = f.copy()
    for j in range(9):
        # Bounce-back
        fc_bb[idx[j]] = f[j].copy()
        # Propagation
        f_propogation[j] = cp.roll(f_propogation[j], ex[j], axis=1)
        f_propogation[j] = cp.roll(f_propogation[j], -ey[j], axis=0)
        f = f_propogation * (solid_node_in_path == 0.) + fc_bb * (solid_node_in_path == 1.)        

    return f


def apply_forcing(rho, ux, uy, fx=cp.float32(0), fy=cp.float32(0), omega=cp.float32(1.0)):
    ux += (fx * rho) / ((rho + 1e-17) ** 2 * omega)
    uy += (fy * rho) / ((rho + 1e-17) ** 2 * omega)
    return ux, uy


def test_lb():


    # Track memory before execution
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    print(f"Before execution: Free GPU memory: {free_mem / 1e6:.2f} MB out of {total_mem / 1e6:.2f} MB")
    
    lx = 1000
    ly = 2000
    ts = 500000
    gx = cp.float32(1e-6)

    # Initialize problem setup
    rowlength = ly
    collength = lx
    rho = cp.zeros((rowlength, collength), dtype=cp.float32)
    ux = cp.zeros((rowlength, collength), dtype=cp.float32)
    uy = cp.zeros((rowlength, collength), dtype=cp.float32)
    obstacles = cp.zeros((rowlength, collength), dtype=cp.float32)
    obstacles[0, :] = 1
    obstacles[-1, :] = 1

    f = cp.zeros((9, rowlength, collength), dtype=cp.float32)
    f[1:5, :, :] = cp.float32(1./9.)
    f[5:9, :, :] = cp.float32(1./36.)
    f[0, :, :] = cp.float32(4./9.) 

    ex = cp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=cp.float32)
    ey = cp.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=cp.float32)
    es = cp.float32(1. / 3.**0.5)
    w = cp.array([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.], dtype=cp.float32)

    for t in range(ts):
        # Compute macro quantities
        rho, ux, uy = compute_macro_quantities(f, obstacles, ex, ey)   

        # Apply forcing
        ux, uy = apply_forcing(rho, ux, uy, fx=gx)

        # Collide
        f = collision(f, rho, ux, uy, obstacles, ex, ey, es, w)

        # Stream
        f = propogation(f, ex, ey, obstacles)

    # Track memory after execution
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    print(f"After execution: Free GPU memory: {free_mem / 1e6:.2f} MB out of {total_mem / 1e6:.2f} MB")

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Compute elapsed time

    print("MLUPS: " ,(lx*ly*ts*1e-6)/elapsed_time)
    # Move data back to CPU for visualization
    plt.figure(1)
    plt.quiver(ux.get(), uy.get())  # Convert to NumPy before plotting
    plt.savefig("velocity_field.png", dpi=300)

    # Plot velocity profile
    ulb = ux[:, int(lx/2)].get()  # Convert to NumPy before plotting
    plt.figure(2)
    plt.plot(ulb)
    plt.savefig("velocity_profile.png", dpi=300)


test_lb()
