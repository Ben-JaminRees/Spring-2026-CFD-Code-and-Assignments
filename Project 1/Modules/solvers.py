"""
solvers.py
----------
This module assembles the tridiagonal matrix and sets up the linalg solver 
for the 1D FVM discretization.
"""

import numpy as np
from scipy.linalg import solve_banded

def build_implicit_matrix(M: int, r: float) -> np.ndarray:
    """
    Constructs the LHS (implicit) matrix in banded format.
    
    Format for scipy.linalg.solve_banded:
    Row 0: Upper diagonal [0, c1, c2, ...]
    Row 1: Main diagonal [b1, b2, b3, ...]
    Row 2: Lower diagonal [a1, a2, a3, ..., 0]

    Inputs
    ------
    M : int
        Number of spatial grid points (control volumes).
    r : float
        The diffusion number, defined as r = (Delta x) / (Delta y^2).

    Returns
    -------
    np.ndarray
        A 3 x M array representing the banded matrix for the implicit scheme.  
    """

    # Initialize a 3 x M array for the diagonals
    A = np.zeros((3, M))

    # 1. Main diagonal (Row 1)
    A[1, :] = 1.0 + 2.0 * r
    A[1, 0] = 1.0 + 6.0 * r # Left boundary correction
    A[1, -1] = 1.0 + 6.0 * r # Right boundary correction

    # 2. Upper diagonal (Row 0)
    A[0, 1:] = -r
    A[0, 1] = -2.0 * r # Left boundary correction

    # 3. Lower diagonal (Row 2)
    A[2, :-1] = -r
    A[2, -2] = -2.0 * r # Right boundary correction

    return A

def construct_rhs(u_n: np.ndarray, r: float, dx: float) -> np.ndarray:
    """
    Assembles the RHS vector 'b' for the system Au=b.

    Includes the explicit Crank-Nicolson terms and the sources term (2 * dx).

    Inputs
    ------
    u_n : np.ndarray
        The current time step values (M x 1).
    r : float
        The diffusion number, defined as r = (Delta x) / (Delta y^2).
    dx : float
        The temporal spacing.

    Returns
    -------
    np.ndarray
        The RHS vector 'b' for the system Au=b.
    """
    M = len(u_n)
    b = np.zeros(M)
    source_term = 2.0 * dx

    # Interior nodes (i=1 to M-2 in 0-based indexing)
    for i in range (1, M - 1):
        b[i] = r * u_n[i - 1] + (1.0 - 2.0 * r) * u_n[i] + r * u_n[i + 1] + source_term

    # Left boundary (i=0)
    b[0] = (1.0 - 6.0 * r) * u_n[0] + 2.0 * r * u_n[1] + source_term

    # Right boundary (i=M-1)
    b[-1] = 2.0 * r * u_n[-2] + (1.0 - 6.0 * r) * u_n[-1] + source_term

    return b

def step_solver(A_banded: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves the linear system Au=b for the next time step values.

    Inputs
    ------
    A_banded : np.ndarray
        The banded matrix representing the LHS of the system.
    b : np.ndarray
        The RHS vector for the system.

    Returns
    -------
    np.ndarray
        The solution vector for the next time step.
    """
    # (1, 1) indicates 1 lower and 1 upper diagonal
    return solve_banded((1, 1), A_banded, b)