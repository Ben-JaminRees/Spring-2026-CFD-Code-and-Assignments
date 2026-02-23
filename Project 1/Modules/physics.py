"""
physics.py
----------
This module defines the physical parameters of the system and models the analytical solution to the 
parabolic equation being solved: du/dx - 2(d^2u/dy^2) = 2.
"""

import numpy as np

def get_bn_coefficient(n: int) -> float:
    """
    Computes the bn coefficient for the Fourier series solution to the PDE.
    
    Parameters
    ----------
    n : int
        The index of the Fourier coefficient to compute.
    
    Returns
    -------
    float
        The value of the bn coefficient for the given n, using the formula:
        B_n = [2 / (n*pi)^3] * [(-1)^n - 1]
    """
    if n % 2 == 0:
        return 0.0
    else:
        return 2.0 / (n * np.pi)**3 * ((-1)**n - 1)
    
def analytical_solution(y: np.ndarray, x: float, n_terms: int = 100) -> np.ndarray:
    """
    Computes the analytical solution to the PDE at a given time x and spatial points y, using a Fourier series expansion.
    
    u(x,y) = SteadyState(y) + Transient(x,y)

    Parameters
    ----------
    y : np.ndarray
        The spatial points at which to evaluate the solution.
    x : float
        The time at which to evaluate the solution.
    n_terms : int, optional
        The number of terms in the Fourier series expansion to compute (default is 100).
    
    Returns
    -------
    np.ndarray
        The analytical solution evaluated at the given spatial points and time.
    """
    # 1. Steady state part: f(y) = y/2 - y^2/2
    steady_state = 0.5 * y * (1.0 - y)

    # 2. Transient part: Sum of B_n + sin(n*pi*y) * exp(-2*(n*pi)^2 * x)
    transient = np.zeros_like(y)
    for n in range(1, n_terms + 1):
        bn = get_bn_coefficient(n)
        eigenvalue = n * np.pi
        spatial_mode = np.sin(eigenvalue * y)
        temporal_decay = np.exp(-2.0 * eigenvalue**2 * x)

        transient += bn * spatial_mode * temporal_decay 
    return steady_state + transient

def initial_condition(y: np.ndarray) -> np.ndarray:
    """
    Computes the initial condition for the PDE at time x=0, which is given by:
    u(0,y) = 0

    Parameters
    ----------
    y : np.ndarray
        The spatial points at which to evaluate the initial condition.
    
    Returns
    -------
    np.ndarray
        The initial condition at the given spatial points.
    """
    return np.zeros_like(y)