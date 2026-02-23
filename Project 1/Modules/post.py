"""
post.py
-------
This module contains functions for post-processing the numerical solution, including error analysis and visualization.
Its purpose is to compare the numerical solution obtained from the finite volume method with the analytical solution derived from the Fourier series expansion.
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_error(u_numerical: np.ndarray, u_analytical: np.ndarray) -> float:
    """
    Computes the L2 norm of the error between the numerical and analytical solutions.

    Inputs
    ------
    u_numerical : np.ndarray
        The numerical solution array at the final time step.
    u_analytical : np.ndarray
        The analytical solution array evaluated at the same spatial points and final time.

    Returns
    -------
    float
        The L2 norm of the error, defined as sqrt(mean((u_numerical - u_analytical)^2)).
    """
    error = np.sqrt(np.mean((u_numerical - u_analytical)**2))
    return error

def plot_profiles(y: np.ndarray,
                  profiles: list[tuple[float, np.ndarray, np.ndarray]],
                  save_path: str = "u_profiles.pdf"):
    """
    Generates a comparison plot for u profiles at different x-stations.

    Inputs
    ------
    y : np.ndarray
        The spatial grid points (y-coordinates).
    profiles : list[tuple[float, np.ndarray, np.ndarray]]
        A list of tuples (x_position, u_numerical, u_analytical) for each profile to be plotted.
    save_path : str
        The file path where the plot will be saved.
    """
    plt.figure(figsize=(8, 6))

    # Iterate through the saved stations
    for x_val, u_num, u_ana in profiles:
        color = plt.cm.viridis(x_val / profiles[-1][0]) # Color based on x-progress

        # Plot numercial as points and analytical as a line
        plt.plot(u_num, y, 'o', markersize=4, label=f'Num (x={x_val})', color=color)
        plt.plot(u_ana, y, '-', linewidth=1.5, label=f'Ana (x={x_val})', color=color)

    plt.xlabel(r'$u(y)$')
    plt.ylabel(r'$y$')
    plt.title('u Profiles at Different Time Steps: Numerical vs. Analytical')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save with high resolution for the report
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved successfully to {save_path}")

def plot_convergence(dx_list: list[float], errors: list[float]):
    """
    Plots error vs step size on a log-log scale to check for 2nd order accuracy.

    Inputs
    ------
    dx_list : list[float]
        A list of step sizes (Delta x) used in the convergence study.
    errors : list[float]
        A list of corresponding L2 errors for each step size.
    """
    plt.figure(figsize=(6, 5))
    plt.loglog(dx_list, errors, '-ok', label='Computed Error')

    # Reference line for 2nd order slope (Error proportional to dx^2)
    plt.loglog(dx_list, [dx**2 for dx in dx_list], '--', label='2nd Order Reference')

    plt.xlabel(r'Step Size $\Delta x$')
    plt.ylabel('L2 Error')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()
    

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_3d_evolution(y: np.ndarray, 
                      all_results: list, 
                      x_range: tuple = (0.0, 0.2), 
                      save_path: str = "isometric_flow.pdf"):
    """
    Creates a 3D isometric plot of u(y) evolving over time (x) to visualize the flow development.

    Inputs
    ------
    y : np.ndarray
        The spatial grid points (y-coordinates).
    all_results : list
        A list of tuples (x_position, u_profile) for each time step saved during the simulation.
    x_range : tuple
        The range of x (time) to include in the plot, defined as (x_min, x_max).
    save_path : str
        The file path where the plot will be saved.
    """
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- 1. Filter the X-Window ---
    # Only keep data where x is within the user-defined x_range
    filtered = [res for res in all_results if x_range[0] <= res[0] <= x_range[1]]
    
    x_coords = np.array([res[0] for res in filtered])
    u_values = np.array([res[1] for res in filtered])
    X, Y = np.meshgrid(x_coords, y)
    Z = u_values.T

    # --- 2. The "Skin" (Transparent Surface) ---
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2, linewidth=0, antialiased=True)

    # --- 3. The "Skeleton" (Progression of Curves) ---
    # cstride=0 means no lines along the y-direction (only x-stations)
    # rstride controls how many x-curves are drawn
    ax.plot_wireframe(X, Y, Z, color='black', alpha=0.5, rstride=0, cstride=20, linewidth=0.8)

    # --- 4. Aesthetics ---
    ax.view_init(elev=20, azim=-150)
    ax.set_box_aspect((2, 1, 0.8)) # Stretch the x-axis for a "tunnel" feel
    
    ax.set_xlabel('Time $x$', labelpad=15)
    ax.set_ylabel('$y$', labelpad=15)
    ax.set_zlabel('$u$', labelpad=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()