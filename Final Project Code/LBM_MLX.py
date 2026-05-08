import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. THE LBM D2Q9 CONSTANTS (PYTHON & MLX HYBRID)
# ---------------------------------------------------------
# standard Python arrays for iteration control and integer shifts
c_py = [
    [ 0,  0], [ 1,  0], [-1,  0], [ 0,  1], [ 0, -1], 
    [ 1,  1], [-1,  1], [-1, -1], [ 1, -1]
]
opposite = [0, 2, 1, 4, 3, 7, 8, 5, 6]

# Convert to MLX Arrays
cx = mx.array([vec[0] for vec in c_py], dtype=mx.float32)
cy = mx.array([vec[1] for vec in c_py], dtype=mx.float32)
w = mx.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=mx.float32)

# ---------------------------------------------------------
# 1.5. STATIC BOUNDARY MASKS
# ---------------------------------------------------------
Nx, Ny = 201, 201

# Build masks in NumPy and cast to MLX to avoid in-place compilation errors
left_mask_np = np.zeros((Nx, Ny), dtype=bool)
left_mask_np[0, :] = True
left_mask = mx.array(left_mask_np)

right_mask_np = np.zeros((Nx, Ny), dtype=bool)
right_mask_np[-1, :] = True
right_mask = mx.array(right_mask_np)

bottom_mask_np = np.zeros((Nx, Ny), dtype=bool)
bottom_mask_np[:, 0] = True
bottom_mask = mx.array(bottom_mask_np)

top_mask_np = np.zeros((Nx, Ny), dtype=bool)
top_mask_np[:, -1] = True
top_mask = mx.array(top_mask_np)

# ---------------------------------------------------------
# 2. THE CORE ENGINE FUNCTIONS (MLX NATIVE)
# ---------------------------------------------------------

def update_macroscopics(f):
    rho = mx.sum(f, axis=0)
    # Broadcasting cx and cy across the grid dimensions
    u = mx.sum(f * cx[:, None, None], axis=0) / rho
    v = mx.sum(f * cy[:, None, None], axis=0) / rho
    return rho, u, v

def calculate_equilibrium(rho, u, v):
    # Fully vectorized calculation
    cu = u[None, :, :] * cx[:, None, None] + v[None, :, :] * cy[:, None, None]
    u_sqr = u**2 + v**2
    f_eq = w[:, None, None] * rho[None, :, :] * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*u_sqr[None, :, :])
    return f_eq

def apply_collision(f, f_eq, tau):
    return f - (1.0 / tau) * (f - f_eq)

def apply_streaming(f):
    f_list = []
    for i in range(9):
        # MLX roll requires integer shifts. roll sequentially in X then Y.
        f_i = mx.roll(f[i], shift=c_py[i][0], axis=0)
        f_i = mx.roll(f_i, shift=c_py[i][1], axis=1)
        f_list.append(f_i)
    # Re-stack into a single tensor
    return mx.stack(f_list, axis=0)

def apply_boundaries(f, f_pre_stream, lid_velocity):
    # Unpack to a list to update specific directions without slice mutation
    f_list = [f[i] for i in range(9)]
    
    # 1. Left Wall
    for i in [1, 5, 8]:
        f_list[i] = mx.where(left_mask, f_pre_stream[opposite[i]], f_list[i])
    # 2. Right Wall
    for i in [2, 6, 7]:
        f_list[i] = mx.where(right_mask, f_pre_stream[opposite[i]], f_list[i])
    # 3. Bottom Wall
    for i in [3, 5, 6]:
        f_list[i] = mx.where(bottom_mask, f_pre_stream[opposite[i]], f_list[i])
        
    # 4. Top Wall (Lid) + Momentum
    rho_wall = 1.0
    for i in [4, 7, 8]:
        f_opp = f_pre_stream[opposite[i]]
        momentum = 6.0 * w[i] * rho_wall * (cx[i] * lid_velocity)
        f_list[i] = mx.where(top_mask, f_opp + momentum, f_list[i])
        
    return mx.stack(f_list, axis=0)

# ---------------------------------------------------------
# 2.5. THE JIT COMPILER
# ---------------------------------------------------------
@mx.compile
def lbm_step(f, rho, u, v, tau, lid_velocity):
    """
    Fuses the entire time-step into a single optimized Metal shader.
    """
    f_eq = calculate_equilibrium(rho, u, v)
    f = apply_collision(f, f_eq, tau)
    f_pre_stream = f
    f = apply_streaming(f)
    f = apply_boundaries(f, f_pre_stream, lid_velocity)
    rho, u, v = update_macroscopics(f)
    return f, rho, u, v

# ---------------------------------------------------------
# 3. THE EXECUTION LOOP
# ---------------------------------------------------------
max_iter = 200000     
lid_velocity = 0.1      
tolerance = 1e-6        

reynolds_numbers = [100, 400, 1000]
lbm_data = {}       

print("Starting Apple Metal MLX Solver...")

for Re in reynolds_numbers:
    print(f"\nSimulating Re = {Re}...")
    
    nu = (lid_velocity * (Ny - 1)) / Re
    tau = 3.0 * nu + 0.5
    
    rho = mx.ones((Nx, Ny), dtype=mx.float32)
    u = mx.zeros((Nx, Ny), dtype=mx.float32)
    v = mx.zeros((Nx, Ny), dtype=mx.float32)
    
    # Initialize Lid
    u_np = np.zeros((Nx, Ny))
    u_np[:, -1] = lid_velocity
    u = mx.array(u_np, dtype=mx.float32)
    
    f = calculate_equilibrium(rho, u, v)
    
    u_prev = u
    residual_history = []
    
    for step in range(max_iter):
        f, rho, u, v = lbm_step(f, rho, u, v, tau, lid_velocity)
        
        # Check convergence and force graph evaluation
        if step % 100 == 0 and step > 0:
            # mx.eval forces the GPU to calculate everything queued up to this point
            mx.eval(f, rho, u, v)
            
            max_error = mx.max(mx.abs(u - u_prev)) / lid_velocity
            max_error_val = max_error.item() # Pull the scalar back to Python
            
            residual_history.append(max_error_val)
            u_prev = u
            
            if step % 5000 == 0:
                print(f"  Iteration {step}: Error = {max_error_val:.2e}")
                
            if max_error_val < tolerance:
                print(f"  --> CONVERGED at iteration {step}!")
                break
                
    print(f"Extraction complete for Re = {Re}.\n")
    
    # Convert MLX arrays back to NumPy for the Matplotlib code below
    u_cpu = np.array(u)
    v_cpu = np.array(v)
    
    lbm_data[Re] = {
        'u_center': u_cpu[Nx//2, :] / lid_velocity,
        'v_center': v_cpu[:, Ny//2] / lid_velocity,
        'u_2d': u_cpu, 
        'v_2d': v_cpu,
        'residuals': residual_history
    }
    
print("All simulations finished. Handing data to Matplotlib...")

# ---------------------------------------------------------
# 4. VISUALIZATION: STREAMLINES AND CONTOURS
# ---------------------------------------------------------
print("Generating contour and streamline plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Lid-Driven Cavity: Velocity Magnitude and Streamlines', fontsize=16)

X, Y = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))

for idx, Re in enumerate(reynolds_numbers):
    ax = axes[idx]
    
    U_plot = lbm_data[Re]['u_2d'].T
    V_plot = lbm_data[Re]['v_2d'].T
    
    speed = np.sqrt(U_plot**2 + V_plot**2) / lid_velocity
    
    contour = ax.contourf(X, Y, speed, levels=50, cmap='viridis')
    ax.streamplot(X, Y, U_plot, V_plot, color='white', density=2.5, linewidth=1.0, arrowsize=1)
    
    ax.set_title(f'$Re =$ {Re}')
    ax.set_xlabel('$X / L$')
    if idx == 0:
        ax.set_ylabel('$Y / L$')
    
    ax.set_aspect('equal')

cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
cbar.set_label('$|V| / U_{lid}$')

plt.savefig('LDC_Streamlines.pdf', dpi=300, bbox_inches='tight')
print("Plot saved as LDC_Streamlines.pdf.")

# ---------------------------------------------------------
# 5. VALIDATION: LBM vs. GHIA (1982) BENCHMARK
# ---------------------------------------------------------
print("Generating Ghia validation plots...")

ghia_y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
ghia_x = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])

ghia_u = {
    100: np.array([1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000]),
    400: np.array([1.0000, 0.7584, 0.6844, 0.6176, 0.5589, 0.2901, 0.1148, -0.0272, -0.1148, -0.1712, -0.3273, -0.2430, -0.1461, -0.1034, -0.0927, -0.0819, 0.0000]),
    1000: np.array([1.0000, 0.6593, 0.5749, 0.5112, 0.4660, 0.3330, 0.1872, 0.0570, -0.0608, -0.1065, -0.2781, -0.3829, -0.2973, -0.2222, -0.2020, -0.1811, 0.0000])
}

ghia_v = {
    100: np.array([0.0000, -0.0591, -0.0739, -0.0886, -0.1031, -0.1691, -0.2245, -0.2453, 0.0545, 0.1753, 0.1751, 0.1608, 0.1232, 0.1089, 0.1009, 0.0923, 0.0000]),
    400: np.array([0.0000, -0.1215, -0.1566, -0.1925, -0.2285, -0.2383, -0.4499, -0.3860, 0.0519, 0.3017, 0.3020, 0.2812, 0.2297, 0.2092, 0.1971, 0.1836, 0.0000]),
    1000: np.array([0.0000, -0.2139, -0.2767, -0.3371, -0.3919, -0.5155, -0.4267, -0.3202, 0.0253, 0.3224, 0.3308, 0.3710, 0.3263, 0.3035, 0.2901, 0.2749, 0.0000])
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Lattice Boltzmann vs. Ghia (1982) Benchmark', fontsize=16)

colors = {100: 'blue', 400: 'green', 1000: 'red'}
markers = {100: 'o', 400: 's', 1000: '^'}

y_lbm = np.linspace(0, 1, Ny)
x_lbm = np.linspace(0, 1, Nx)

for Re in reynolds_numbers:
    ax1.plot(lbm_data[Re]['u_center'], y_lbm, color=colors[Re], label=f'LBM Re={Re}', linewidth=2)
    ax1.scatter(ghia_u[Re], ghia_y, edgecolors=colors[Re], facecolors='none', marker=markers[Re], s=50, label=f'Ghia Re={Re}')
    
    ax2.plot(x_lbm, lbm_data[Re]['v_center'], color=colors[Re], linewidth=2)
    ax2.scatter(ghia_x, ghia_v[Re], edgecolors=colors[Re], facecolors='none', marker=markers[Re], s=50)

ax1.set_title('U-Velocity along Vertical Centerline (x = 0.5)')
ax1.set_xlabel('$u / U_{lid}$')
ax1.set_ylabel('$y / L$')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

ax2.set_title('V-Velocity along Horizontal Centerline (y = 0.5)')
ax2.set_xlabel('$x / L$')
ax2.set_ylabel('$v / U_{lid}$')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.savefig('LDC_Validation_Profiles.pdf', dpi=300, bbox_inches='tight')
print("Plot saved as LDC_Validation_Profiles.pdf.")

# ---------------------------------------------------------
# 6. VISUALIZATION: CONVERGENCE / RESIDUAL HISTORY
# ---------------------------------------------------------
print("Generating Convergence History plot...")

plt.figure(figsize=(8, 6))
plt.title('LBM Solver Convergence History', fontsize=14)

for Re in reynolds_numbers:
    if 'residuals' in lbm_data[Re] and len(lbm_data[Re]['residuals']) > 0:
        plt.plot(lbm_data[Re]['residuals'], label=f'Re = {Re}', linewidth=2)

plt.yscale('log')
plt.xlabel('Convergence Check Iteration (x100 steps)')
plt.ylabel('Max Velocity Error (Normalized)')
#plt.axhline(y=tolerance, color='black', linestyle='--', label='Tolerance Threshold')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

plt.savefig('LDC_Convergence.pdf', dpi=300, bbox_inches='tight')
print("Plot saved as LDC_Convergence.pdf.")
plt.show()