"""
Ben Rees
02/22/2026
CFD Project 1
main.py
-------
Main execution script for the 1D Channel Flow development simulation.
Solves the parabolic momentum equation using a Crank-Nicolson FVM scheme.

Case Parameters:
- Domain: y in [0, 1]
- Scheme: Crank-Nicolson (2nd order)
- Linear Solver: TDMA (via scipy.linalg.solve_banded)
"""

import numpy as np
import physics as phys
import solvers as sol
import post

def run_simulation(M: int = 50, x_final: float = 1.0, dx: float = 0.001):
    # 1. Grid Generation
    dy = 1.0 / M
    y = np.linspace(dy/2, 1.0 - dy/2, M) # Cell-centered nodes
    r = dx / (dy**2)
    
    # 2. Initialization
    u = phys.initial_condition(y)
    x = 0.0
    
    # Stations to save for plotting (as requested by the report)
    save_stations = [0.01, 0.1, 1.0]
    results = []

    # NEW: Storage for the "Crazy" 3D Plot
    all_history = [] 
    save_interval = 20  # Save every 20 steps to keep the 3D plot smooth but fast
    step_count = 0
    
    # 3. Pre-compute constant Implicit Matrix
    A_banded = sol.build_implicit_matrix(M, r)
    
    # 4. Time-Marching Loop
    print(f"Starting simulation with r={r:.4f}...")
    while x < x_final:
        # Check if we should save current profile before stepping
        for station in save_stations:
            if abs(x - station) < dx/2:
                u_ana = phys.analytical_solution(y, x)
                results.append((round(x, 2), u.copy(), u_ana))

        #Save data for 3D plot every 'save_interval' steps
        if step_count % save_interval == 0:
            all_history.append((x, u.copy()))
        
        # Build RHS and Solve
        b = sol.construct_rhs(u, r, dx)
        u = sol.step_solver(A_banded, b)
        
        x += dx
    
    # 5. Final Comparison & Plotting
    post.plot_profiles(y, results, save_path="u_comparison.pdf")

    # Plot 2: The "Portfolio" Plot (3D)
    post.plot_3d_evolution(y, all_history, x_range=(0.0, 0.25), save_path="isometric_flow.pdf")

    # Final error check
    u_final_ana = phys.analytical_solution(y, x_final)
    error = post.calculate_error(u, u_final_ana)
    print(f"Simulation complete. L2 Error at x=1.0: {error:.2e}")

if __name__ == "__main__":
    # Standard parameters for the project
    run_simulation(M=50, x_final=1.05, dx=0.001)